"""
Test script for validating audio embedding quality.

This script tests whether your 4-stage embedding pipeline 
(Basic Pitch → Encodec → VQ-VAE → CLAP) produces good representations
by using a CRNN decoder to transcribe audio to tablature.
"""

import torch
import numpy as np
from models.audio_embedding_pipeline import GuitarAudioEmbeddingPipeline
from models.embedding_validation_decoder import EmbeddingValidationSystem


def create_synthetic_guitar_audio(duration=2.0, sample_rate=22050):
    """Create synthetic guitar audio for testing."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create a simple guitar-like chord (Em: E2, G3, B3)
    frequencies = [82.41, 196.0, 246.94]  # E2, G3, B3
    audio = np.zeros_like(t)
    
    for freq in frequencies:
        # Add fundamental + harmonics
        note = (
            np.sin(2 * np.pi * freq * t) +
            0.3 * np.sin(2 * np.pi * 2 * freq * t) +
            0.1 * np.sin(2 * np.pi * 3 * freq * t)
        )
        
        # Add envelope
        envelope = np.exp(-t * 0.8)  # Decay
        audio += note * envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.7
    
    return torch.tensor(audio, dtype=torch.float32)


def test_embedding_validation():
    """Test the embedding validation system."""
    print("Testing Audio Embedding Validation System")
    print("=" * 50)
    
    # Create synthetic test audio
    print("\n1. Creating synthetic guitar audio...")
    audio = create_synthetic_guitar_audio(duration=3.0)
    print(f"   Audio shape: {audio.shape}")
    print(f"   Duration: {len(audio) / 22050:.2f} seconds")
    
    # Initialize embedding pipeline
    print("\n2. Initializing embedding pipeline...")
    embedding_pipeline = GuitarAudioEmbeddingPipeline(
        sample_rate=22050,
        device='cpu'  # Use CPU for testing
    )
    print("   ✓ Embedding pipeline initialized")
    
    # Initialize validation system
    print("\n3. Initializing validation system...")
    validation_system = EmbeddingValidationSystem(
        embedding_pipeline=embedding_pipeline,
        device='cpu'
    )
    print("   ✓ Validation system initialized")
    
    # Test embedding extraction
    print("\n4. Testing embedding extraction...")
    with torch.no_grad():
        # Test if embeddings can be extracted
        embedding_outputs = embedding_pipeline(audio.unsqueeze(0))
        embeddings = embedding_outputs['embeddings']
        print(f"   ✓ Embeddings extracted: {embeddings.shape}")
        print(f"   ✓ Embedding stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
        
        # Check individual components
        if 'pitch_features' in embedding_outputs:
            print(f"   ✓ Basic Pitch features: {embedding_outputs['pitch_features'].shape}")
        if 'discrete_codes' in embedding_outputs:
            print(f"   ✓ VQ-VAE codes: {embedding_outputs['discrete_codes'].shape}")
        if 'semantic_features' in embedding_outputs:
            print(f"   ✓ CLAP semantic features: {embedding_outputs['semantic_features'].shape}")
    
    # Test validation (transcription)
    print("\n5. Testing embedding validation (transcription)...")
    results = validation_system.validate_embeddings(
        audio, 
        return_intermediate=True
    )
    
    # Display results
    print(validation_system.format_results(results))
    
    # Check if reasonable results
    guitar_notes = results['guitar_notes']
    metrics = results['embedding_quality_metrics']
    
    if metrics['notes_detected'] > 0:
        print(f"\n   ✓ SUCCESS: Detected {metrics['notes_detected']} notes")
        print("   Your embeddings appear to contain musical information!")
    else:
        print(f"\n   ⚠ WARNING: No notes detected")
        print("   Your embeddings may not be capturing musical content effectively.")
        
    return True


def test_embedding_quality_across_components():
    """Test which embedding components contribute most to transcription quality."""
    print("\n\nTesting Individual Embedding Components")
    print("=" * 50)
    
    # This would test ablations:
    # 1. Basic Pitch only
    # 2. Basic Pitch + Encodec  
    # 3. Basic Pitch + Encodec + VQ-VAE
    # 4. Full pipeline (+ CLAP)
    
    print("(This function would test embedding component ablations)")
    print("- Basic Pitch features alone")
    print("- + Encodec compression")  
    print("- + VQ-VAE discrete tokens")
    print("- + CLAP semantic understanding")
    print("\nFor now, focus on getting the full pipeline working first.")


def main():
    """Main test function."""
    print("Audio Embedding Quality Validation")
    print("==================================")
    print("This tests if your 4-stage embedding pipeline produces")
    print("representations suitable for guitar transcription.\n")
    
    # Test basic validation
    success = test_embedding_validation()
    
    if success:
        print("\n" + "=" * 50)
        print("NEXT STEPS:")
        print("1. If notes detected: Your embeddings are working!")
        print("2. If no notes: Check embedding pipeline components")
        print("3. Add real audio files for more realistic testing")
        print("4. Compare results with/without different embedding stages")
        print("5. Train the decoder on GuitarSet data to see learning capacity")
    else:
        print("\n" + "=" * 50)
        print("TROUBLESHOOTING NEEDED:")
        print("1. Check if Basic Pitch and CLAP models have pre-trained weights")
        print("2. Verify VQ-VAE and Encodec implementations")
        print("3. Test each embedding component individually")
        print("4. Check audio preprocessing pipeline")


if __name__ == "__main__":
    main()