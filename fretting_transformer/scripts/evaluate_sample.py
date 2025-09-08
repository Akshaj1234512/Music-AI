#!/usr/bin/env python3
"""
Evaluate Fretting Transformer on Single GuitarSet Excerpt

This script loads a specific GuitarSet excerpt by ID and evaluates it with
the model, showing detailed input/output analysis and saving results to disk.
"""

import sys
import os
import json
import argparse
import torch
import pickle
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.guitarset_loader import GuitarSetLoader
from data.unified_tokenizer import UnifiedFrettingTokenizer
from model.unified_fretting_t5 import create_model_from_tokenizer
from inference.postprocess import FrettingPostProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate model on single GuitarSet excerpt')
    
    # Required arguments
    parser.add_argument('excerpt_id', type=str,
                       help='GuitarSet excerpt ID (e.g., "00_BN1-129-Eb_comp")')
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--guitarset_path', type=str, 
                       default='/data/akshaj/MusicAI/GuitarSet',
                       help='Path to GuitarSet dataset')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str,
                       default='/data/andreaguz/sample_outputs',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_tokens', action='store_true',
                       help='Save detailed token analysis')
    parser.add_argument('--save_audio_info', action='store_true',
                       help='Include audio file paths in output')
    
    # Generation parameters
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum generation length')
    parser.add_argument('--num_beams', type=int, default=4,
                       help='Number of beams for beam search')
    parser.add_argument('--chunk_size_notes', type=int, default=20,
                       help='Number of notes per chunk')
    
    # Post-processing
    parser.add_argument('--apply_postprocessing', action='store_true',
                       help='Apply post-processing to fix pitch accuracy')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='Specific GPU ID to use')
    
    return parser.parse_args()


def setup_device(device_arg, gpu_id=None):
    """Setup compute device."""
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"🔧 Using GPU {gpu_id}")
    
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Device: {device}")
    return device


def load_model_and_tokenizer(model_path=None, device='cuda'):
    """Load model and tokenizer."""
    print("Loading tokenizer...")
    tokenizer = UnifiedFrettingTokenizer()
    
    print("Creating model...")
    model = create_model_from_tokenizer(tokenizer, model_type='paper')
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        
        # Handle different checkpoint formats
        if os.path.isdir(model_path):
            # Hugging Face format
            model_file = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location='cpu')
            else:
                raise FileNotFoundError(f"Model file not found in {model_path}")
        else:
            # Single checkpoint file
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded successfully")
    else:
        print("⚠️  No model specified, using untrained model")
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def convert_notes_to_tokens(notes, tokenizer, chunk_size_notes=20):
    """Convert GuitarSet notes to input/target token sequences."""
    if not notes:
        return [], []
    
    # Sort notes by time
    sorted_notes = sorted(notes, key=lambda x: x.start_time)
    
    # Take first chunk for simplicity
    chunk_notes = sorted_notes[:chunk_size_notes]
    
    # Convert to relative timing
    base_time = chunk_notes[0].start_time
    
    # Create MIDI input tokens
    input_tokens = [tokenizer.config.bos_token]
    target_tokens = [tokenizer.config.bos_token]
    
    current_time = 0
    
    for note in chunk_notes:
        # Calculate relative timing
        note_start = note.start_time - base_time
        note_duration = note.duration
        
        # Add time shift to start if needed
        if note_start > current_time:
            time_shift = int((note_start - current_time) * 960)  # Convert to MIDI ticks
            # Find closest available time shift
            available_shifts = tokenizer.config.time_shifts
            time_shift = min(available_shifts, key=lambda x: abs(x - time_shift))
            
            input_tokens.append(f"TIME_SHIFT<{time_shift}>")
            target_tokens.append(f"TIME_SHIFT<{time_shift}>")
            current_time = note_start
        
        # Add note events
        input_tokens.append(f"NOTE_ON<{note.pitch}>")
        target_tokens.append(f"TAB<{note.string + 1},{note.fret}>")  # Convert to 1-based string
        
        # Add duration
        duration_ticks = int(note_duration * 960)
        duration_ticks = min(available_shifts, key=lambda x: abs(x - duration_ticks))
        
        input_tokens.append(f"TIME_SHIFT<{duration_ticks}>")
        target_tokens.append(f"TIME_SHIFT<{duration_ticks}>")
        
        input_tokens.append(f"NOTE_OFF<{note.pitch}>")
        current_time += note_duration
    
    input_tokens.append(tokenizer.config.eos_token)
    target_tokens.append(tokenizer.config.eos_token)
    
    return input_tokens, target_tokens


def generate_tablature(model, tokenizer, input_tokens, device, max_length=512, num_beams=4):
    """Generate tablature from input tokens."""
    # Convert tokens to IDs
    input_ids = [tokenizer.token_to_id.get(token, tokenizer.unk_token_id) 
                for token in input_tokens]
    
    # Convert to tensor
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Generating with {num_beams} beams, max_length={max_length}")
    
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    # Convert back to tokens
    predicted_tokens = []
    for token_id in generated[0]:
        if token_id.item() in tokenizer.id_to_token:
            predicted_tokens.append(tokenizer.id_to_token[token_id.item()])
        else:
            predicted_tokens.append(tokenizer.config.unk_token)
    
    return predicted_tokens


def analyze_results(input_tokens, target_tokens, predicted_tokens, predicted_pp_tokens=None):
    """Analyze and compare results."""
    analysis = {
        'input_length': len(input_tokens),
        'target_length': len(target_tokens),
        'predicted_length': len(predicted_tokens),
        'input_notes': len([t for t in input_tokens if t.startswith('NOTE_ON')]),
        'target_tabs': len([t for t in target_tokens if t.startswith('TAB')]),
        'predicted_tabs': len([t for t in predicted_tokens if t.startswith('TAB')]),
    }
    
    if predicted_pp_tokens:
        analysis['predicted_pp_length'] = len(predicted_pp_tokens)
        analysis['predicted_pp_tabs'] = len([t for t in predicted_pp_tokens if t.startswith('TAB')])
    
    # Calculate basic accuracy
    if analysis['target_tabs'] > 0 and analysis['predicted_tabs'] > 0:
        # Simple token-level accuracy
        min_tabs = min(analysis['target_tabs'], analysis['predicted_tabs'])
        target_tab_tokens = [t for t in target_tokens if t.startswith('TAB')][:min_tabs]
        predicted_tab_tokens = [t for t in predicted_tokens if t.startswith('TAB')][:min_tabs]
        
        matches = sum(1 for t, p in zip(target_tab_tokens, predicted_tab_tokens) if t == p)
        analysis['tab_accuracy'] = matches / min_tabs if min_tabs > 0 else 0
    else:
        analysis['tab_accuracy'] = 0
    
    return analysis


def save_results(excerpt_id, excerpt, input_tokens, target_tokens, predicted_tokens, 
                predicted_pp_tokens, analysis, output_dir, save_tokens=False):
    """Save evaluation results to disk."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results dictionary
    results = {
        'excerpt_id': excerpt_id,
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'style': excerpt.style,
            'mode': excerpt.mode,
            'player_id': excerpt.player_id,
            'total_notes': len(excerpt.notes),
            'duration_seconds': excerpt.notes[-1].start_time + excerpt.notes[-1].duration - excerpt.notes[0].start_time if excerpt.notes else 0
        },
        'analysis': analysis,
        'sample_predictions': {
            'input_tokens_sample': input_tokens[:10],
            'target_tokens_sample': target_tokens[:10],
            'predicted_tokens_sample': predicted_tokens[:10]
        }
    }
    
    if predicted_pp_tokens:
        results['sample_predictions']['predicted_pp_tokens_sample'] = predicted_pp_tokens[:10]
    
    # Save detailed tokens if requested
    if save_tokens:
        results['full_tokens'] = {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'predicted_tokens': predicted_tokens
        }
        if predicted_pp_tokens:
            results['full_tokens']['predicted_pp_tokens'] = predicted_pp_tokens
    
    # Save main results
    results_file = os.path.join(output_dir, f'{excerpt_id}_evaluation.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}")
    
    # Save human-readable summary
    summary_file = os.path.join(output_dir, f'{excerpt_id}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"GuitarSet Excerpt Evaluation: {excerpt_id}\n")
        f.write(f"{'='*50}\n\n")
        
        f.write(f"Metadata:\n")
        f.write(f"  Style: {excerpt.style}\n")
        f.write(f"  Mode: {excerpt.mode}\n") 
        f.write(f"  Player: {excerpt.player_id}\n")
        f.write(f"  Total Notes: {len(excerpt.notes)}\n\n")
        
        f.write(f"Analysis:\n")
        for key, value in analysis.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write(f"\nSample Tokens (first 10):\n")
        f.write(f"Input:     {input_tokens[:10]}\n")
        f.write(f"Target:    {target_tokens[:10]}\n") 
        f.write(f"Predicted: {predicted_tokens[:10]}\n")
        
        if predicted_pp_tokens:
            f.write(f"Post-Proc: {predicted_pp_tokens[:10]}\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    
    return results_file, summary_file


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("="*60)
    print(f"EVALUATING GUITARSET EXCERPT: {args.excerpt_id}")
    print("="*60)
    
    # Setup
    device = setup_device(args.device, args.gpu_id)
    
    # Load excerpt
    print(f"\nLoading excerpt: {args.excerpt_id}")
    loader = GuitarSetLoader(args.guitarset_path)
    
    try:
        excerpt = loader.load_excerpt_by_id(args.excerpt_id)
        print(f"✅ Loaded excerpt with {len(excerpt.notes)} notes")
        print(f"   Style: {excerpt.style}, Mode: {excerpt.mode}, Player: {excerpt.player_id}")
    except FileNotFoundError:
        print(f"❌ Excerpt {args.excerpt_id} not found!")
        print("\nAvailable excerpts:")
        available = loader.get_all_excerpt_ids()[:10]
        for excerpt_id in available:
            print(f"  {excerpt_id}")
        return
    
    # Convert to tokens
    print(f"\nConverting to tokens...")
    tokenizer = UnifiedFrettingTokenizer()
    input_tokens, target_tokens = convert_notes_to_tokens(
        excerpt.notes, tokenizer, args.chunk_size_notes
    )
    
    print(f"Input tokens: {len(input_tokens)}")
    print(f"Target tokens: {len(target_tokens)}")
    
    # Load model and generate
    print(f"\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    
    print(f"\nGenerating tablature...")
    predicted_tokens = generate_tablature(
        model, tokenizer, input_tokens, device, args.max_length, args.num_beams
    )
    
    print(f"Generated {len(predicted_tokens)} tokens")
    
    # Apply post-processing if requested
    predicted_pp_tokens = None
    if args.apply_postprocessing:
        print(f"\nApplying post-processing...")
        postprocessor = FrettingPostProcessor(tokenizer)
        
        predicted_pp_tokens, pp_stats = postprocessor.process_tablature(
            input_tokens, predicted_tokens
        )
        
        print(f"Post-processing stats: {pp_stats}")
    
    # Analyze results
    print(f"\nAnalyzing results...")
    analysis = analyze_results(input_tokens, target_tokens, predicted_tokens, predicted_pp_tokens)
    
    print(f"Analysis:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    print(f"\nSaving results...")
    results_file, summary_file = save_results(
        args.excerpt_id, excerpt, input_tokens, target_tokens, 
        predicted_tokens, predicted_pp_tokens, analysis, 
        args.output_dir, args.save_tokens
    )
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results: {results_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()