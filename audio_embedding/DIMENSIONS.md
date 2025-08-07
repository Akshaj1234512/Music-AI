# Audio Embedding Pipeline - Dimension Guide

## Input Dimensions

### Raw Audio Input
- **Format**: `[batch_size, time_samples]`
- **Time**: **FLEXIBLE** - Any audio length works
- **Sample rate**: 22,050 Hz (can be configured)
- **Example**: 5 seconds = 110,250 samples

### Constraints
- Minimum length: ~0.5 seconds (11,025 samples) for meaningful features
- Maximum length: Limited only by memory
- Batch size: Flexible, limited by GPU memory

## Output Dimensions

### Time Frame Calculation
```
time_frames = audio_length // hop_length
hop_length = 512 (default)

Examples:
- 0.5 sec (11,025 samples) → 21 frames
- 1.0 sec (22,050 samples) → 43 frames  
- 5.0 sec (110,250 samples) → 215 frames
```

### Component Outputs

#### 1. Basic Pitch (Real Spotify Model)
- **Features**: `[batch, time_frames, 440]`  
- **440** = 88 onset + 264 contour + 88 note (pre-trained Spotify model)

#### 2. Meta Encodec (HuggingFace)
- **Embeddings**: `[batch, time_frames, 8]` (8 codebooks as embeddings)
- **Codes**: `[batch, time_frames, 8]` (8 RVQ stages)
- **Model**: `facebook/encodec_24khz` - Meta's official pretrained weights
- **Bandwidth**: 6.0 kbps target compression
- **Resampling**: 22kHz → 24kHz automatically handled

#### 3. Kena VQ-VAE (Refactored for Multimodal)
- **Embeddings**: `[batch, time_frames, 64]` - For fusion layer
- **Discrete tokens**: `[batch, time_frames]` - For pattern learning
- **Token range**: 0-511 (512 codebook entries)
- **Commitment loss**: Scalar - For VQ training
- **NO direct transcription** - Moved to separate audio decoder

#### 4. CLAP (HuggingFace Music Model)
- **Global embeddings**: `[batch, 768]` (no time dimension)
- **Model**: `laion/larger_clap_music` - pretrained on music datasets
- **Input resampling**: 22kHz → 48kHz automatically handled
- **Capabilities**: Zero-shot classification, semantic understanding

### Final Pipeline Output
- **Fused embeddings**: `[batch, time_frames, 768]`
- All component outputs projected to 768 dims

## Why These Dimensions?

### Variable Time Input
- **Real-world flexibility**: Songs have different lengths
- **Chunking support**: Can process long audio in segments
- **Frame-based**: Each frame represents ~23ms of audio

### Fixed Feature Dimensions
- **440 (Basic Pitch)**: Real Spotify model (88 onset + 264 contour + 88 note)
- **8 (Encodec)**: Meta's RVQ codebooks (8 stages for 6.0 kbps)
- **64 (Kena VQ-VAE)**: Efficient discrete representation (8 attention heads)
- **88 (Transcription)**: Piano range (A0-C8) covers all guitar pitches
- **768 (CLAP/Final)**: Standard transformer embedding size, music-pretrained

### Discrete Tokens & Transcription
- **512 codebook size**: Good balance for guitar patterns
- **Per-frame tokens**: Enables sequence modeling
- **Integer values**: Efficient storage and lookup
- **88 piano keys**: Complete pitch range for guitar transcription
- **Dual predictions**: Onset + frame for complete note modeling

## Practical Implications

### Memory Usage (per second of audio)
```
Input: 22,050 × 4 bytes = 86 KB
Basic Pitch: 43 × 440 × 4 = 76 KB (updated for 440 dims)
Encodec: 43 × 8 × 4 = 1.4 KB (HuggingFace Meta model)
Kena VQ-VAE: 43 × 64 × 4 = 11 KB
Audio Decoder: 43 × 88 × 2 × 4 = 30 KB (onset + frame)
CLAP: 768 × 4 = 3 KB (music-pretrained)
Final: 43 × 768 × 4 = 129 KB

Total: ~336 KB per second
```

### Batch Processing
- **Recommendation**: Use consistent lengths within batch
- **Padding**: Required for batching different lengths
- **Efficiency**: Larger batches = better GPU utilization

### Downstream Compatibility
- **768-dim embeddings**: Compatible with standard transformer models
- **Frame-aligned**: Easy temporal alignment with video (30fps)
- **Discrete tokens**: Enable modern sequence modeling (GPT-style)

## Code Examples

### Processing Variable Length Audio
```python
# Single short clip
short_audio = torch.randn(1, 22050)  # 1 second
short_output = pipeline(short_audio)
# Output: [1, 43, 768]

# Batch of different lengths (requires padding)
audio_batch = [
    torch.randn(22050),    # 1 sec
    torch.randn(44100),    # 2 sec
    torch.randn(33075),    # 1.5 sec
]
# Pad to max length
max_len = max(a.shape[0] for a in audio_batch)
padded = torch.stack([F.pad(a, (0, max_len - a.shape[0])) for a in audio_batch])
output = pipeline(padded)
# Output: [3, 86, 768] (based on 2 sec max)
```

### Chunking Long Audio
```python
def process_long_audio(audio, chunk_size=110250, overlap=11025):
    """Process long audio in overlapping chunks."""
    chunks = []
    for i in range(0, len(audio), chunk_size - overlap):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
    
    # Process all chunks
    batch = torch.stack(chunks)
    embeddings = pipeline(batch)
    
    # Merge overlapping regions (simple average)
    # ... (implementation depends on use case)
    return embeddings
```

## Transcription System Dimensions (Refactored)

### Audio Transcription Decoder (Primary)
- **Input**: Fused embeddings `[batch, time_frames, 768]`
- **Architecture**: CRNN (CNN → GRU → Multi-head outputs)
- **Onset probabilities**: `[batch, time_frames, 88]` (piano range)
- **Frame probabilities**: `[batch, time_frames, 88]` (piano range)
- **Confidence scores**: `[batch, time_frames]`
- **Purpose**: Audio-only transcription + embedding validation

### Future: Multimodal Decoder
- **Input**: Fused audio+video embeddings `[batch, time_frames, fused_dim]`
- **Architecture**: Enhanced decoder with cross-attention
- **Output**: Same as audio decoder but with video-enhanced accuracy
- **Target**: 89% accuracy (vs 87% audio-only)

### Note Event Output
- **Note list**: Variable length list of (pitch, onset_frame, offset_frame)
- **Pitch range**: 21-108 (A0 to C8)
- **Time resolution**: ~23ms per frame

### Tab Assignment Output
- **Guitar notes**: List of GuitarNote objects with:
  - String: 0-5 (high E to low E)
  - Fret: 0-24
  - Technique: String identifier

### Final Tablature
- **ASCII format**: 6 lines (strings) × variable length
- **Time resolution**: Configurable (default 0.25s)

## Architecture Changes Summary

### NEW: Integrated Kena VQ-VAE
- **Single Component**: VQ-VAE now handles both discrete tokens AND transcription
- **Direct Predictions**: Onset/frame predictions come directly from VQ-VAE decoder
- **No Separate Decoder**: Eliminates need for separate pitch decoder

### System Dimensions
- **Input**: Flexible time dimension, fixed sample rate
- **Embeddings**: Frame-based with fixed feature dimensions (768)
- **Transcription**: 88-key piano range per frame (direct from Kena VQ-VAE)
- **Tab Output**: 6 strings × 24 frets possibilities  
- **Discrete Tokens**: 512 VQ-VAE codebook for patterns
- **Memory**: ~340KB per second for embeddings + transcription