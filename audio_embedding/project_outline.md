# Multimodal Guitar Transcription Implementation Guide

> **IMPLEMENTATION NOTE**: This document describes the full multimodal (audio + video) system targeting 89% accuracy.
> 
> **Current Status**: We have implemented the audio-only baseline first:
> - Audio-only target: 87% accuracy (Kena AI benchmark)
> - Video fusion can be added later for the additional 2% improvement
> - See `models/guitar_transcription_system.py` for current audio-only implementation

## Project Overview

This guide provides comprehensive implementation details for building a multimodal guitar transcription system that combines audio and video processing to detect notes, chords, and playing techniques (slides, bends, hammer-ons, pull-offs).

### Target Performance
- **89% transcription accuracy** (with audio + video)
- **87% transcription accuracy** (audio-only - IMPLEMENTED)
- **<10ms latency** for real-time processing
- Technique detection: slides, bends, hammer-ons, pull-offs, vibrato
- Output tokens: `Note_E4`, `Slide`, `Bend`, `Hammer_On`, etc.

## Core Architecture Decision

**Selected Stack**: Hierarchical 4-stage pipeline
1. **Basic Pitch** (Spotify) - Initial transcription
2. **Meta Encodec** - Audio compression/encoding
3. **VQ-VAE** (Kena AI inspired) - Structured representation
4. **CLAP** - Semantic understanding and contrastive pretraining

## Stage 1: Basic Pitch Implementation

### Repository and Resources
- **GitHub**: https://github.com/spotify/basic-pitch
- **Paper**: https://engineering.atspotify.com/2022/06/meet-basic-pitch
- **PyTorch Implementation**: https://github.com/gudgud96/basic-pitch-torch

### Key Architecture Details
```python
# Basic Pitch Architecture
- Input: Audio waveform (22.05 kHz)
- Processing: Harmonic Constant-Q Transform (CQT)
- Model: Shallow CNN with ~17,000 parameters
- Outputs: 
  - Contour (continuous pitch) - for bends/slides
  - Note (discrete pitches) - for note detection
  - Onset (attack timing) - for technique detection
```

### Implementation Notes
- **Memory footprint**: <20MB
- **Real-time capable**: Processes faster than audio playback
- **Multi-output structure**: Access intermediate representations before final predictions

### Embedding Extraction
```python
# Access intermediate layers for embeddings
# Key layers to extract from:
1. Harmonic stacking layer output
2. Intermediate CNN feature maps  
3. Pre-output dense layer activations
```

## Stage 2: Meta Encodec Integration

### Repository and Resources
- **GitHub**: https://github.com/facebookresearch/encodec
- **Paper**: https://arxiv.org/abs/2210.13438 (High Fidelity Neural Audio Compression)

### Key Concepts
- **Residual Vector Quantization (RVQ)**: 8 codebook stages
- **Compression**: 40% better than traditional codecs
- **Bitrates**: 1.5-24 kbps with high quality

### Implementation Details
```python
# Encodec Architecture
- Encoder: Convolutional with strided convolutions
- Quantizer: 8-stage RVQ, 1024 codes per stage
- Decoder: Transposed convolutions
- Loss: Reconstruction + Perceptual + Adversarial
```

### Integration Points
- Process Basic Pitch outputs through Encodec
- Use compressed representations for efficient storage/transmission
- Extract codebook indices as discrete tokens

## Stage 3: Kena AI VQ-VAE Implementation

### Resources
- **Kena AI Article**: https://medium.com/autonomous-agents/kenas-artificial-intelligence-is-the-most-powerful-and-accurate-music-neural-engine-214f4f524e60
- **Original VQ-VAE Paper**: https://arxiv.org/abs/1711.00937
- **Efficient Training**: https://arxiv.org/html/2507.10547v1 (Quantize-then-Rectify)

### Kena AI's Complete VQ-VAE System
**Important**: Kena AI IS a VQ-VAE-based music transcription system, not a separate component. Their innovation is applying VQ-VAE with dual-objective loss directly to music transcription.

### Kena AI's Dual-Loss Approach
```python
# Loss Function Components
L_total = L_onset + L_frame + L_commitment

# L_onset: Note beginning detection
# L_frame: Frame-level reconstruction  
# L_commitment: VQ codebook learning

# Contrastive Pretraining Loss
L_contrastive = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
# τ = temperature parameter (typically 0.07)
```

### Architecture Specifications
- **Encoder**: 4-layer CNN, stride 2, channels [512, 512, 512, 512]
- **Codebook**: 512 codes, 64 dimensions
- **Decoder**: Mirror of encoder with transposed convolutions
- **Training time**: 22 hours on single GPU (modern approach)

### Guitar-Specific Adaptations
- Modify codebook to capture guitar-specific patterns
- Add string-aware pooling layers
- Include technique-specific heads for classification

## Stage 4: CLAP Integration

### Resources
- **GitHub**: https://github.com/LAION-AI/CLAP
- **Paper**: https://arxiv.org/abs/2206.04769
- **Pretrained Models**: Available via HuggingFace

### Key Implementation Details
```python
# CLAP Architecture
- Audio Encoder: HTS-AT or PANN backbone
- Text Encoder: RoBERTa or BERT
- Temperature: τ = 0.07 (learnable)
- Batch Size: 2048 for contrastive learning
```

### Guitar-Specific Text Pairs
```python
# Example training pairs for guitar
audio_texts = [
    "clean electric guitar playing E major chord",
    "distorted guitar with palm muting",
    "guitar slide from 5th to 7th fret",
    "hammer-on technique on high E string",
    "vibrato on bent note"
]
```

### Style Embeddings
Extract style embeddings as per "Assessing the Alignment of Audio Representations with Timbre Similarity Ratings" (https://arxiv.org/html/2507.07764v1):
- Use Gram matrix of feature activations
- Extract from multiple layers
- Combine for multi-scale representation

## Cross-Attention Fusion with MediaPipe

### Resources
- **MediaPipe Hands**: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
- **Multimodal Fusion Survey**: https://arxiv.org/html/2411.17040v1

### Fusion Architecture
```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, audio_dim=768, video_dim=63, hidden_dim=512):
        # audio_dim: CLAP embedding size
        # video_dim: 21 landmarks × 3 coordinates
        
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
    def forward(self, audio_emb, video_emb):
        # Project to common space
        audio_hidden = self.audio_proj(audio_emb)
        video_hidden = self.video_proj(video_emb)
        
        # Cross-attention (audio attends to video)
        fused, _ = self.cross_attn(
            query=audio_hidden,
            key=video_hidden,
            value=video_hidden
        )
        return fused
```

### Temporal Alignment
- Sample audio at 44.1kHz, process in 1024-sample windows
- Video at 30fps → align every ~1470 audio samples
- Use sliding window for smooth transitions

## DDSP for Guitar-Specific Processing

### Resources
- **DDSP Guitar Amp Paper**: https://arxiv.org/abs/2408.11405
- **General DDSP Review**: https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1284100/full

### Implementation
```python
# DDSP Components for Guitar
1. Harmonic Oscillator: Model string vibrations
2. Filtered Noise: Fret noise, pick attack
3. Reverb: Room acoustics
4. Amplifier Model: Tone shaping

# Training Loss
L = L_spectral + λ_f0 * L_f0 + λ_loudness * L_loudness
```

## Token Generation Pipeline

### Token Vocabulary
```python
TOKEN_VOCAB = {
    # Note tokens (per string-fret combination)
    'Note_E2_0': 0,    # Open low E string
    'Note_E2_1': 1,    # Low E string, 1st fret
    # ... up to 24th fret for 6 strings
    
    # Technique tokens
    'Slide_Up': 200,
    'Slide_Down': 201,
    'Hammer_On': 202,
    'Pull_Off': 203,
    'Bend_Half': 204,
    'Bend_Full': 205,
    'Vibrato': 206,
    
    # Special tokens
    'PAD': 300,
    'START': 301,
    'END': 302
}
```

### Token Generation Logic
```python
def generate_tokens(audio_features, video_features):
    # 1. Basic Pitch → initial note predictions
    notes = basic_pitch_model(audio_features)
    
    # 2. VQ-VAE → discrete representations
    vq_codes = vq_vae.encode(audio_features)
    
    # 3. Cross-attention fusion
    fused_features = cross_attention(vq_codes, video_features)
    
    # 4. Technique classification
    techniques = technique_classifier(fused_features)
    
    # 5. Combine into token sequence
    tokens = merge_notes_and_techniques(notes, techniques)
    
    return tokens
```

## Training Strategy

### Data Requirements
1. **GuitarSet**: 360 tracks, hexaphonic recordings
   - Paper: https://arxiv.org/html/2408.08653v1
2. **DadaGP**: 33,967 symbolic guitar tabs
   - Used for pairwise likelihood estimation
3. **Custom Dataset**: Needed for technique labels

### Progressive Training Schedule
```python
# Phase 1: Pretrain on piano (MAESTRO dataset)
epochs_1 = 50
lr_1 = 1e-3

# Phase 2: Fine-tune on guitar
epochs_2 = 100  
lr_2 = 1e-4

# Phase 3: Technique-specific fine-tuning
epochs_3 = 50
lr_3 = 1e-5
```

### Loss Weighting
```python
total_loss = (
    1.0 * note_loss +
    0.5 * technique_loss +
    0.3 * timing_loss +
    0.2 * contrastive_loss
)
```

## Performance Optimization

### Real-time Processing Requirements
- **Latency budget**: 10ms total
  - Audio encoding: 3ms
  - Cross-attention: 4ms  
  - Token generation: 3ms

### Memory Optimization
- Use mixed precision training (fp16)
- Gradient checkpointing for large models
- Quantize models for deployment (INT8)

### GPU Requirements
- Training: Single A100 (40GB) or 2×V100 (32GB)
- Inference: RTX 3060 or better for real-time

## Evaluation Metrics

### Transcription Metrics
```python
# From "A Data-Driven Methodology for Guitar Tablature"
# Paper: https://arxiv.org/pdf/2204.08094.pdf

- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)  
- F1-score: 2 * (Precision * Recall) / (Precision + Recall)
- TDR (Tablature Disambiguation Rate): Correct string assignments
```

### Technique Detection Metrics
- Per-technique precision/recall
- Temporal alignment accuracy (±50ms window)
- Confusion matrix for technique types

## Known Challenges and Solutions

### Challenge 1: Duplicate Pitch Errors
**Problem**: Same pitch on different strings
**Solution**: Use inhibition loss from "Data-Driven Methodology" paper
```python
L_inhibition = Σ_pairs w(i,j) * activation_i * activation_j
# w(i,j) = inhibition weight between string-fret pairs
```

### Challenge 2: Technique Timing
**Problem**: Techniques occur between notes
**Solution**: Use overlapping windows and post-processing smoothing

### Challenge 3: Effect Pedal Interference  
**Problem**: Distortion masks harmonic content
**Solution**: DDSP preprocessing to model effects

## Deployment Considerations

### Model Serving
```python
# FastAPI endpoint example
@app.post("/transcribe")
async def transcribe(audio: bytes, video: bytes):
    # 1. Preprocess inputs
    audio_tensor = preprocess_audio(audio)
    video_tensor = extract_hand_landmarks(video)
    
    # 2. Run inference
    tokens = model.generate_tokens(audio_tensor, video_tensor)
    
    # 3. Post-process
    tablature = tokens_to_tablature(tokens)
    
    return {"tablature": tablature}
```

### Edge Deployment
- Use ONNX for cross-platform compatibility
- Implement frame-based processing for streaming
- Cache frequently used embeddings

## References for Implementation

1. **Basic Pitch Implementation Details**
   - Source code analysis: https://github.com/spotify/basic-pitch/blob/main/basic_pitch/models.py
   - Model architecture: https://github.com/spotify/basic-pitch/blob/main/basic_pitch/constants.py

2. **VQ-VAE Best Practices**
   - Efficient training: https://arxiv.org/html/2507.10547v1
   - Codebook collapse prevention: Use exponential moving average updates

3. **Cross-Attention Implementation**
   - Multimodal fusion survey: https://arxiv.org/html/2411.17040v1
   - Attention bottlenecks: Reduce keys/values by factor of 4

4. **Guitar-Specific Resources**
   - GuitarSet dataset: https://zenodo.org/record/1492449
   - TENT technique detection: https://transactions.ismir.net/articles/10.5334/tismir.23

## Code Organization

```
guitar_transcription/
├── models/
│   ├── basic_pitch.py
│   ├── encodec_wrapper.py
│   ├── vq_vae.py
│   ├── clap_encoder.py
│   └── cross_attention.py
├── data/
│   ├── dataset.py
│   ├── augmentation.py
│   └── tokenizer.py
├── training/
│   ├── train.py
│   ├── losses.py
│   └── metrics.py
├── inference/
│   ├── real_time.py
│   └── batch_process.py
└── utils/
    ├── audio_processing.py
    └── video_processing.py
```

## Next Steps

1. **Implement Basic Pitch baseline** - Verify 89.7% F-measure on GuitarSet
2. **Add VQ-VAE encoding** - Train on guitar audio for 22 hours
3. **Integrate CLAP** - Use pretrained model, fine-tune on guitar descriptions
4. **Build fusion module** - Implement cross-attention with MediaPipe
5. **Create technique classifier** - Binary classifiers for each technique
6. **Optimize for real-time** - Profile and optimize bottlenecks
7. **Evaluate on test set** - Compare with Guitar2Tabs, Kena AI

This guide provides all necessary technical details for implementation. Each component has been validated in research with specific performance metrics. Following this architecture should achieve the target 89% accuracy with real-time performance.