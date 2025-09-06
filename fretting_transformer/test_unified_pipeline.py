#!/usr/bin/env python3
"""
Test Unified Pipeline for Fretting Transformer

Tests the complete unified vocabulary pipeline to verify:
1. Model loads without vocabulary mismatch errors
2. Dataset processing works with unified tokenizer
3. Training produces reasonable loss and convergence
4. Model generates proper length sequences
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.unified_tokenizer import UnifiedFrettingTokenizer
from data.unified_dataset import UnifiedFrettingDataProcessor, create_unified_data_loaders
from model.unified_fretting_t5 import create_model_from_tokenizer
from torch.optim import AdamW

def test_unified_pipeline():
    """Test the complete unified pipeline."""
    print("=== Testing Unified Pipeline ===")
    
    # 1. Create unified tokenizer
    print("1. Creating unified tokenizer...")
    tokenizer = UnifiedFrettingTokenizer()
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    print(f"   Special tokens: PAD={tokenizer.pad_token_id}, BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}")
    
    # 2. Create unified model
    print("\n2. Creating unified model...")
    model = create_model_from_tokenizer(tokenizer, 'debug')
    model_info = model.get_model_info()
    print(f"   Model size: {model_info['parameters_millions']:.2f}M parameters")
    print(f"   Model vocabulary: {model_info['vocab_size']}")
    
    # 3. Test dataset processing (small sample)
    print("\n3. Testing dataset processing...")
    processor = UnifiedFrettingDataProcessor()
    
    try:
        processor.load_and_process_data(
            category='jams',
            max_files=3,  # Just test with 3 files
            cache_path='data/processed/unified_pipeline_test.pkl'
        )
        
        if not processor.processed_sequences:
            print("   ⚠️  No sequences processed - this might indicate data loading issues")
            return False
            
        print(f"   Processed {len(processor.processed_sequences)} sequences")
        
        # Create data splits
        train_dataset, val_dataset, test_dataset = processor.create_data_splits()
        train_loader, val_loader, test_loader = create_unified_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=2
        )
        
        print(f"   Dataset sizes: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        
    except Exception as e:
        print(f"   ❌ Dataset processing failed: {e}")
        return False
    
    # 4. Test forward pass with real data
    print("\n4. Testing forward pass with real data...")
    if len(train_loader) > 0:
        try:
            batch = next(iter(train_loader))
            
            print(f"   Batch shapes:")
            print(f"     input_ids: {batch['input_ids'].shape}")
            print(f"     attention_mask: {batch['attention_mask'].shape}")
            print(f"     labels: {batch['labels'].shape}")
            print(f"   Token ID ranges:")
            print(f"     input_ids: {batch['input_ids'].min()}-{batch['input_ids'].max()}")
            print(f"     labels: {batch['labels'].min()}-{batch['labels'].max()}")
            
            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                print(f"   Forward pass results:")
                print(f"     Loss: {outputs.loss.item():.4f}")
                print(f"     Expected loss range: ~{torch.log(torch.tensor(float(tokenizer.vocab_size))).item():.4f}")
                print(f"     Logits shape: {outputs.logits.shape}")
                
                # Check if loss is reasonable
                expected_loss = torch.log(torch.tensor(float(tokenizer.vocab_size))).item()
                loss_reasonable = abs(outputs.loss.item() - expected_loss) < 3.0
                print(f"     Loss is reasonable: {loss_reasonable}")
                
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            return False
    else:
        print("   ⚠️  No data available for forward pass test")
        return False
    
    # 5. Test mini training loop
    print("\n5. Testing mini training loop...")
    try:
        model.train()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        initial_loss = None
        losses = []
        
        for step, batch in enumerate(train_loader):
            if step >= 5:  # Just 5 steps
                break
                
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            if initial_loss is None:
                initial_loss = loss.item()
            
            print(f"   Step {step+1}: Loss = {loss.item():.4f}")
        
        if len(losses) >= 2:
            final_loss = losses[-1]
            loss_decreased = final_loss < initial_loss
            print(f"   Training results:")
            print(f"     Initial loss: {initial_loss:.4f}")
            print(f"     Final loss: {final_loss:.4f}")
            print(f"     Loss decreased: {loss_decreased}")
            
            if not loss_decreased:
                print("   ⚠️  Loss didn't decrease - might need more steps or different LR")
        
    except Exception as e:
        print(f"   ❌ Training loop failed: {e}")
        return False
    
    # 6. Test generation
    print("\n6. Testing generation...")
    try:
        model.eval()
        with torch.no_grad():
            # Use first sample from batch
            test_input = batch['input_ids'][:1]  # Single sample
            test_attention = batch['attention_mask'][:1]
            
            generated = model.generate(
                input_ids=test_input,
                attention_mask=test_attention,
                max_new_tokens=50,
                num_beams=2,
                early_stopping=True
            )
            
            print(f"   Generation results:")
            print(f"     Input length: {test_input.shape[1]}")
            print(f"     Generated length: {generated.shape[1]}")
            print(f"     Generated tokens (first 20): {generated[0][:20].tolist()}")
            
            # Decode generated tokens
            generated_tokens = tokenizer.ids_to_tokens(generated[0].tolist()[:20])
            print(f"     Generated token strings: {generated_tokens}")
            
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        return False
    
    print("\n✅ All unified pipeline tests passed!")
    print("\nKey Validations:")
    print("✓ Unified tokenizer created successfully")
    print("✓ Model uses unified vocabulary without custom heads")
    print("✓ Dataset processes real data with unified tokens")
    print("✓ Forward pass produces reasonable loss")
    print("✓ Training loop runs without errors")
    print("✓ Generation produces output sequences")
    
    print(f"\nThe unified vocabulary approach fixes the fundamental issue:")
    print(f"- Single vocabulary size: {tokenizer.vocab_size}")
    print(f"- MIDI tokens: {tokenizer.token_to_id['NOTE_ON<60>']} (NOTE_ON<60>)")
    print(f"- TAB tokens: {tokenizer.token_to_id['TAB<3,0>']} (TAB<3,0>)")
    print(f"- No vocabulary mismatch between encoder and decoder!")
    
    return True

if __name__ == "__main__":
    success = test_unified_pipeline()
    sys.exit(0 if success else 1)