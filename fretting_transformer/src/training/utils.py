"""
Training Utilities for Fretting Transformer

Helper functions for training, monitoring, and debugging.
"""

import os
import json
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_model_memory(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """
    Estimate model memory usage.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, seq_len)
        
    Returns:
        Dictionary with memory estimates in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    # Estimate activation memory (rough approximation)
    batch_size, seq_len = input_shape
    # Assume float32 for activations
    activation_memory = batch_size * seq_len * 4  # 4 bytes per float32
    
    # Convert to MB
    param_mb = param_size / (1024 * 1024)
    buffer_mb = buffer_size / (1024 * 1024)
    activation_mb = activation_memory / (1024 * 1024)
    
    total_mb = param_mb + buffer_mb + activation_mb
    
    return {
        'parameters_mb': param_mb,
        'buffers_mb': buffer_mb,
        'activations_mb': activation_mb,
        'total_mb': total_mb
    }


def save_training_metrics(metrics: Dict[str, List[float]], save_path: str):
    """
    Save training metrics to JSON file.
    
    Args:
        metrics: Dictionary of metric lists
        save_path: Path to save file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        save_path: Optional[str] = None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses  
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add min validation loss annotation
    min_val_epoch = val_losses.index(min(val_losses)) + 1
    min_val_loss = min(val_losses)
    plt.annotate(f'Min Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_epoch}',
                xy=(min_val_epoch, min_val_loss),
                xytext=(min_val_epoch + len(epochs) * 0.1, min_val_loss + max(val_losses) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_rate_schedule(learning_rates: List[float], 
                               save_path: Optional[str] = None):
    """
    Plot learning rate schedule.
    
    Args:
        learning_rates: List of learning rates per epoch
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 4))
    
    epochs = range(1, len(learning_rates) + 1)
    plt.plot(epochs, learning_rates, 'g-', linewidth=2)
    
    plt.title('Learning Rate Schedule (Adafactor)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


class TrainingMonitor:
    """
    Monitor training progress and detect issues.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.losses = []
        self.gradients = []
        
    def add_loss(self, loss: float):
        """Add a loss value."""
        self.losses.append(loss)
        
    def add_gradient_norm(self, grad_norm: float):
        """Add gradient norm value."""
        self.gradients.append(grad_norm)
        
    def is_loss_exploding(self, threshold: float = 10.0) -> bool:
        """Check if loss is exploding."""
        if len(self.losses) < 2:
            return False
            
        recent_losses = self.losses[-self.window_size:]
        if len(recent_losses) < 2:
            return False
            
        # Check if loss increased significantly
        loss_ratio = recent_losses[-1] / recent_losses[0]
        return loss_ratio > threshold
    
    def is_loss_plateauing(self, threshold: float = 0.01, min_steps: int = 20) -> bool:
        """Check if loss has plateaued."""
        if len(self.losses) < min_steps:
            return False
            
        recent_losses = self.losses[-min_steps:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Check if relative standard deviation is small
        rel_std = loss_std / loss_mean if loss_mean > 0 else float('inf')
        return rel_std < threshold
    
    def get_gradient_status(self) -> str:
        """Get gradient status (normal, vanishing, exploding)."""
        if not self.gradients:
            return "unknown"
            
        recent_grads = self.gradients[-self.window_size:]
        avg_grad = np.mean(recent_grads)
        
        if avg_grad < 1e-7:
            return "vanishing"
        elif avg_grad > 10.0:
            return "exploding"
        else:
            return "normal"
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.losses:
            return {}
            
        return {
            'total_steps': len(self.losses),
            'current_loss': self.losses[-1],
            'min_loss': min(self.losses),
            'max_loss': max(self.losses),
            'loss_trend': 'improving' if len(self.losses) > 1 and self.losses[-1] < self.losses[0] else 'degrading',
            'is_exploding': self.is_loss_exploding(),
            'is_plateauing': self.is_loss_plateauing(),
            'gradient_status': self.get_gradient_status()
        }


def debug_model_outputs(model, tokenizer, sample_batch, max_samples: int = 3):
    """
    Debug model outputs by examining generated sequences.
    
    Args:
        model: FrettingT5Model
        tokenizer: FrettingTokenizer
        sample_batch: Sample batch from dataloader
        max_samples: Maximum samples to examine
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Move batch to device
        batch = {k: v.to(device) for k, v in sample_batch.items()}
        
        # Get model outputs
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        print("=== Model Debug Information ===")
        print(f"Batch size: {batch['input_ids'].shape[0]}")
        print(f"Sequence length: {batch['input_ids'].shape[1]}")
        print(f"Loss: {outputs.loss.item():.4f}")
        
        # Examine first few samples
        for i in range(min(max_samples, batch['input_ids'].shape[0])):
            print(f"\n--- Sample {i+1} ---")
            
            # Input tokens
            input_ids = batch['input_ids'][i]
            input_tokens = tokenizer.ids_to_tokens(input_ids.cpu().tolist(), 'input')
            print("Input (first 10 tokens):", input_tokens[:10])
            
            # Target tokens
            target_ids = batch['labels'][i]
            target_tokens = tokenizer.ids_to_tokens(target_ids.cpu().tolist(), 'output')
            print("Target (first 10 tokens):", target_tokens[:10])
            
            # Predicted tokens
            pred_ids = predictions[i]
            pred_tokens = tokenizer.ids_to_tokens(pred_ids.cpu().tolist(), 'output')
            print("Predicted (first 10 tokens):", pred_tokens[:10])
            
            # Calculate token-level accuracy
            valid_positions = (target_ids != -100)  # Ignore padded positions
            if valid_positions.sum() > 0:
                token_accuracy = (pred_ids[valid_positions] == target_ids[valid_positions]).float().mean()
                print(f"Token accuracy: {token_accuracy.item():.4f}")


if __name__ == "__main__":
    # Test utilities
    print("Testing training utilities...")
    
    # Test seed setting
    set_seed(42)
    print("✓ Seed set")
    
    # Test time formatting
    print(f"Time format test: {format_time(3661)}")  # Should show 1.0h
    
    # Test training monitor
    monitor = TrainingMonitor()
    for i in range(20):
        monitor.add_loss(1.0 / (i + 1))  # Decreasing loss
        monitor.add_gradient_norm(0.1 + 0.01 * i)
    
    summary = monitor.get_training_summary()
    print("Training monitor summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("All utilities tested successfully!")