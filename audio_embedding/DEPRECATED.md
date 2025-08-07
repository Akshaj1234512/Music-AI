# Deprecated Components

This file tracks components that have been superseded but are kept for reference.

## January 2025 - Production-Ready Pretrained Models

### Deprecated: `models/DEPRECATED_basic_pitch_wrapper.py` (formerly `basic_pitch_wrapper.py`)
- **Replaced by**: `models/clean_basic_pitch_wrapper.py` + `preprocessing/extract_basic_pitch_features.py`
- **Reason**: Inefficient file I/O and TensorFlow/PyTorch integration issues
- **New Approach**: Dual strategy for optimal performance
- **Advantages of Replacement**:
  - ✅ Offline preprocessing with official Spotify TensorFlow model (maximum reliability)
  - ✅ Online inference with PyTorch basic-pitch-torch (gradient flow enabled)
  - ✅ No temporary file I/O during training
  - ✅ Clean separation between preprocessing and training pipeline
  - ✅ Faster training iterations (precomputed features)
  - ✅ No TensorFlow/PyTorch memory conflicts during training
- **Status**: Replaced with dual approach for production use

### Deprecated: `models/encodec_wrapper.py`
- **Replaced by**: `models/huggingface_encodec.py`
- **Reason**: Custom Encodec implementation replaced by Meta's official pretrained model
- **Model Change**: Custom RVQ implementation → `facebook/encodec_24khz` 
- **Advantages of Replacement**:
  - ✅ Meta's official pretrained weights (`facebook/encodec_24khz`)
  - ✅ Music and speech optimized (24kHz model)
  - ✅ HuggingFace ecosystem integration with `EncodecFeatureExtractor`
  - ✅ Automatic sample rate handling (22kHz → 24kHz)
  - ✅ Proven compression performance at 6.0 kbps
  - ✅ 8 codebook RVQ stages for efficient discrete representation
  - ✅ Production-ready neural codec with extensive testing
- **Status**: Kept for reference, fully replaced in pipeline

### Deprecated: `models/clap_encoder.py`
- **Replaced by**: `models/huggingface_clap.py`
- **Reason**: Custom CLAP implementation replaced by HuggingFace pretrained music model
- **Model Change**: Custom HTS-AT implementation → `laion/larger_clap_music`
- **Advantages of Replacement**:
  - ✅ Music-specific pretraining (51% GTZAN accuracy)
  - ✅ HuggingFace ecosystem integration
  - ✅ Production-ready with extensive testing
  - ✅ Automatic sample rate handling (22kHz → 48kHz)
  - ✅ Zero-shot audio classification capabilities
- **Status**: Kept for reference, replaced in pipeline

## January 2025 - Real Basic Pitch Integration

### Deprecated: `models/basic_pitch.py`
- **Replaced by**: `models/basic_pitch_wrapper.py` 
- **Reason**: Custom reimplementation replaced by real Spotify Basic Pitch with pre-trained weights
- **Output Dimension Change**: 360 dims → 440 dims (88 onset + 264 contour + 88 note)
- **Advantages of Replacement**:
  - ✅ Pre-trained weights (no training needed)
  - ✅ Research-validated accuracy  
  - ✅ Cross-platform deployment support
  - ✅ Maintained by Spotify team
- **Status**: Kept for reference, not used in pipeline

### API Compatibility
The new `BasicPitchFeatureExtractor` in `basic_pitch_wrapper.py` maintains the same API as the original, ensuring existing code continues to work:

```python
# Old API (deprecated)
from models.basic_pitch import BasicPitchFeatureExtractor

# New API (current)  
from models.basic_pitch_wrapper import BasicPitchFeatureExtractor

# Same usage pattern
extractor = BasicPitchFeatureExtractor(sample_rate=22050)
features = extractor(audio)  # Now returns [batch, time, 440] instead of [batch, time, 360]
```

### Migration Guide
1. **Code**: No changes needed (API compatible)
2. **Dimensions**: Update downstream components expecting 360 → 440 dims
3. **Accuracy**: Expect immediate improvement from pre-trained weights
4. **Deployment**: Consider backend choice (CoreML/TensorFlow/ONNX/TFLite)

## File Removal Policy
Deprecated files are kept in the repository for:
- Historical reference
- Understanding implementation decisions  
- Academic/research purposes
- Potential future alternative implementations

They are not used in the active pipeline and will not receive updates.