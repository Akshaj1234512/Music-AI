"""
Training Pipeline for Fretting Transformer

Implements the training loop with Adafactor optimizer and adaptive learning rate
as specified in the paper. Includes validation, checkpointing, and logging.
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor, AdafactorSchedule
from tqdm import tqdm

from model.fretting_t5 import FrettingT5Model
from data.tokenizer import FrettingTokenizer


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Training hyperparameters
    num_epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Learning rate and optimization
    learning_rate: float = None  # Adaptive with Adafactor
    warmup_steps: int = 1000
    
    # Validation and logging
    validation_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 1000
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    
    # Directories
    output_dir: str = "experiments/checkpoints"
    logging_dir: str = "experiments/logs"
    
    # Model checkpointing
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    
    # Mixed precision training
    use_fp16: bool = False
    
    # Seed for reproducibility
    seed: int = 42


class FrettingTrainer:
    """
    Trainer class for the Fretting Transformer model.
    
    Handles training loop, validation, checkpointing, and logging
    using the Adafactor optimizer as specified in the paper.
    """
    
    def __init__(self,
                 model: FrettingT5Model,
                 tokenizer: FrettingTokenizer,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 config: TrainingConfig):
        
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer (Adafactor with internal LR scheduling)
        self.optimizer = Adafactor(
            self.model.parameters(),
            relative_step=True,    # Let Adafactor handle LR internally
            scale_parameter=True,
            warmup_init=True
        )
        # No external scheduler needed - Adafactor handles it internally
        
        # Set up directories
        self.setup_directories()
        
        # Initialize logging
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Mixed precision training
        if self.config.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def setup_directories(self):
        """Create output directories."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logging_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(self.config.logging_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log training configuration
        self.logger.info("Training Configuration:")
        for key, value in asdict(self.config).items():
            self.logger.info(f"  {key}: {value}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training history and final metrics
        """
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model size: {self.model.get_model_info()['parameters_millions']:.2f}M parameters")
        
        # Set random seed
        torch.manual_seed(self.config.seed)
        
        training_start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            train_loss = self.train_epoch()
            
            # Check if training actually happened
            if train_loss is None:
                self.logger.error(f"Training epoch returned None - no batches processed!")
                self.logger.error(f"Training dataset size: {len(self.train_dataloader.dataset)}")
                self.logger.error(f"Training dataloader batches: {len(self.train_dataloader)}")
                raise RuntimeError("No training batches were processed")
            
            # Validation phase
            val_loss = self.validate()
            
            # Learning rate tracking (Adafactor, relative_step=True supported)
            def _current_lr_from_adafactor(optimizer):
                try:
                    group = optimizer.param_groups[0]
                    # If LR is explicit (e.g., relative_step=False), just use it
                    if group.get("lr", None) is not None:
                        return float(group["lr"])
                    # Otherwise compute the effective LR like Adafactor does
                    p = group["params"][0]
                    state = optimizer.state.get(p, {})
                    # _get_lr needs both the group and the param's state (requires at least one .step() happened)
                    return float(optimizer._get_lr(group, state))  # HF Adafactor private helper
                except Exception:
                    # Fallback to 0.0 if called before any optimizer.step()
                    return 0.0
            
            current_lr = _current_lr_from_adafactor(self.optimizer)
            self.learning_rates.append(current_lr)
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, lr={current_lr:.2e}"
            )
            
            # Save checkpoint
            if (epoch + 1) % (self.config.save_steps // len(self.train_dataloader)) == 0:
                self.save_checkpoint()
            
            # Early stopping check
            if self.early_stopping_check(val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        training_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {training_time / 3600:.2f} hours")
        
        # Load best model if configured
        if self.config.load_best_model_at_end:
            self.load_best_checkpoint()
        
        # Convert any tensors to regular Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if hasattr(obj, 'item'):  # PyTorch tensor
                return obj.item()
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        return {
            'train_losses': convert_to_json_serializable(self.train_losses),
            'val_losses': convert_to_json_serializable(self.val_losses),
            'learning_rates': convert_to_json_serializable(self.learning_rates),
            'final_train_loss': convert_to_json_serializable(self.train_losses[-1] if self.train_losses else None),
            'final_val_loss': convert_to_json_serializable(self.val_losses[-1] if self.val_losses else None),
            'best_val_loss': convert_to_json_serializable(self.best_val_loss),
            'total_training_time': training_time,
            'total_steps': self.global_step
        }
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        epoch_steps = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch + 1}",
            leave=False
        )
        
        for step, batch in enumerate(progress_bar):
            loss = self.training_step(batch)
            total_loss += loss
            epoch_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}: loss={loss:.4f}, "
                    f"lr={self.optimizer.param_groups[0].get('lr', 0):.2e}"
                )
            
            # Validation during training
            if self.global_step % self.config.validation_steps == 0:
                val_loss = self.validate()
                self.logger.info(f"Validation at step {self.global_step}: loss={val_loss:.4f}")
                self.model.train()  # Return to training mode
        
        # Flush leftover gradients if epoch ended mid-accumulation block
        leftover = (self.global_step % self.config.gradient_accumulation_steps) != 0
        if leftover:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        
        avg_loss = total_loss / epoch_steps if epoch_steps > 0 else 0.0
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute one training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Loss value
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        use_amp = self.scaler is not None
        accum_steps = max(1, int(self.config.gradient_accumulation_steps))  # Guard against 0
        
        # Zero gradients at accumulation boundary
        if self.global_step % accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)
        
        # --- Forward pass ---
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = self.model(**batch)
                loss = outputs.loss
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
        
        # Scale loss for gradient accumulation to maintain effective LR
        if accum_steps > 1:
            loss = loss / accum_steps
        
        # --- Backward pass ---
        if use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # --- Step only on accumulation boundary ---
        do_step = ((self.global_step + 1) % accum_steps == 0)
        
        # Optional gradient clipping (only when stepping)
        if do_step and self.config.max_grad_norm:
            if use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        if do_step:
            if use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            # No scheduler.step(): Adafactor (relative_step=True) handles LR internally
        
        # Advance global micro-step counter each batch
        self.global_step += 1
        
        # ALWAYS return a float (unscaled loss for logging)
        return float(loss.detach().item() * accum_steps)
    
    def validate(self) -> float:
        """
        Run validation on the validation set.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps if total_steps > 0 else float('inf')
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def early_stopping_check(self, val_loss: float) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_val_loss - self.config.early_stopping_threshold:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            f"checkpoint-{self.global_step}"
        )
        
        # Save model
        self.model.save_model(checkpoint_dir)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': asdict(self.config)
        }
        
        torch.save(state, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        self.logger.info(f"Checkpoint saved at step {self.global_step}")
        
        # Clean up old checkpoints
        self.cleanup_checkpoints()
    
    def cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoint_dirs = [
            d for d in os.listdir(self.config.output_dir)
            if d.startswith('checkpoint-')
        ]
        
        if len(checkpoint_dirs) > self.config.save_total_limit:
            # Sort by step number and remove oldest
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
            for dir_to_remove in checkpoint_dirs[:-self.config.save_total_limit]:
                checkpoint_path = os.path.join(self.config.output_dir, dir_to_remove)
                import shutil
                shutil.rmtree(checkpoint_path)
                self.logger.info(f"Removed old checkpoint: {dir_to_remove}")
    
    def load_best_checkpoint(self):
        """Load the best checkpoint based on validation loss."""
        if not self.val_losses:
            return
        
        best_epoch = self.val_losses.index(min(self.val_losses))
        # Find corresponding checkpoint (approximate)
        best_step = (best_epoch + 1) * len(self.train_dataloader)
        
        checkpoint_dirs = [
            d for d in os.listdir(self.config.output_dir)
            if d.startswith('checkpoint-')
        ]
        
        if checkpoint_dirs:
            # Find closest checkpoint to best step
            steps = [int(d.split('-')[1]) for d in checkpoint_dirs]
            closest_step = min(steps, key=lambda x: abs(x - best_step))
            best_checkpoint = f"checkpoint-{closest_step}"
            
            checkpoint_path = os.path.join(self.config.output_dir, best_checkpoint)
            if os.path.exists(checkpoint_path):
                self.model = FrettingT5Model.load_model(checkpoint_path)
                self.model.to(self.device)
                self.logger.info(f"Loaded best checkpoint: {best_checkpoint}")
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        # Load model
        self.model = FrettingT5Model.load_model(checkpoint_path)
        self.model.to(self.device)
        
        # Load training state
        state_path = os.path.join(checkpoint_path, 'training_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_val_loss = state['best_val_loss']
            self.patience_counter = state['patience_counter']
            self.train_losses = state['train_losses']
            self.val_losses = state['val_losses']
            self.learning_rates = state['learning_rates']
            
            # Load optimizer state
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            
            self.logger.info(f"Resumed training from step {self.global_step}")


def create_trainer(model: FrettingT5Model,
                  tokenizer: FrettingTokenizer,
                  train_dataloader: DataLoader,
                  val_dataloader: DataLoader,
                  config: Optional[TrainingConfig] = None) -> FrettingTrainer:
    """
    Create a trainer instance.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer instance
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        config: Training configuration
        
    Returns:
        FrettingTrainer instance
    """
    if config is None:
        config = TrainingConfig()
    
    return FrettingTrainer(model, tokenizer, train_dataloader, val_dataloader, config)


if __name__ == "__main__":
    # Test trainer setup
    from model.fretting_t5 import create_model_from_tokenizer
    from data.tokenizer import FrettingTokenizer
    
    # Create dummy components for testing
    tokenizer = FrettingTokenizer()
    model = create_model_from_tokenizer(tokenizer, 'debug')
    
    print("Trainer components created successfully!")
    print(f"Model size: {model.get_model_info()['parameters_millions']:.2f}M parameters")
    
    # Test Adafactor optimizer
    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=True,
        relative_step_size=True,
        warmup_init=False,
        lr=None
    )
    
    print("Adafactor optimizer initialized successfully!")
    print(f"Optimizer state: {len(optimizer.param_groups)} parameter groups")