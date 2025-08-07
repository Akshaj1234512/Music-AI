# Changelog

All notable changes to the Guitar Audio Embedding Pipeline project will be documented in this file.

## [Unreleased]

### January 2025 - Week 4

#### Added - Dual Basic Pitch Strategy
- **Offline Preprocessing** (`preprocessing/extract_basic_pitch_features.py`)
  - Uses official Spotify Basic Pitch (TensorFlow) for maximum reliability
  - Extracts onset, contour, note features from audio datasets offline
  - Saves precomputed features as PyTorch tensors (.pt files)
  - Includes feature manifest and validation for dataset integrity
- **Online Training Pipeline** (`models/clean_basic_pitch_wrapper.py`)
  - Dual-mode wrapper supporting precomputed and real-time inference
  - PyTorch basic-pitch-torch integration for gradient flow when needed
  - No TensorFlow/PyTorch conflicts during training
  - Significant training speedup (no Basic Pitch inference during training)
- **Precomputed Dataset Loader** (`datasets/precomputed_features_dataset.py`)
  - Efficient loading of audio + precomputed Basic Pitch features
  - Automatic feature/audio alignment validation
  - Custom collate function for variable-length sequences
  - Memory-efficient batching with padding masks
- **Pipeline Integration** (`models/audio_embedding_pipeline.py`)
  - Updated to support precomputed Basic Pitch features
  - Maintains backward compatibility for real-time inference
  - Clean separation between feature extraction and training

#### Added - Meta Encodec HuggingFace Integration
- **HuggingFace Encodec Integration** (`models/huggingface_encodec.py`)
  - Meta's official pretrained Encodec model: `facebook/encodec_24khz`
  - Production-ready neural audio codec with RVQ (Residual Vector Quantization)
  - Automatic sample rate handling from 22kHz to 24kHz
  - 8 codebook compression with 6.0 kbps target bandwidth  
  - Proper EncodecFeatureExtractor usage (specialized over AutoProcessor)
  - Replaces custom Encodec wrapper with Meta's pretrained weights
- **Pipeline Integration** (`models/audio_embedding_pipeline.py`)
  - Updated Encodec import to use HuggingFace implementation
  - Corrected embedding dimension from 128 to 8 (n_codebooks)
  - Maintained compression and embedding extraction functionality
- **Complete Production Pipeline**
  - All 4 components now use production-ready pretrained models
  - End-to-end system with Meta Encodec + Spotify Basic Pitch + HuggingFace CLAP
  - No custom audio codec training required

### January 2025 - Week 4

#### Added - Production-Ready Pretrained Models
- **Spotify Basic Pitch Integration** (`models/basic_pitch_wrapper.py`)
  - PyTorch wrapper for official Spotify Basic Pitch with pre-trained weights
  - Cross-platform compatibility (CoreML, TensorFlow, ONNX, TFLite backends)
  - Proper dimension handling: 440 features (88 onset + 264 contour + 88 note)
  - Temporary file handling for Basic Pitch's file-based API
  - Individual output extraction and structured note events
- **HuggingFace CLAP Integration** (`models/huggingface_clap.py`)
  - Music-specific pretrained model: `laion/larger_clap_music`
  - Automatic resampling from 22kHz to 48kHz (CLAP requirement)
  - Semantic embeddings for guitar technique understanding
  - Zero-shot audio classification capabilities
  - Replaces custom CLAP implementation with production-ready model
- **Pipeline Updates** (`models/audio_embedding_pipeline.py`)
  - Updated Basic Pitch import to use real implementation
  - Integrated HuggingFace CLAP replacing custom encoder
  - Added temporal dimension alignment between pipeline components  
  - F.interpolate for aligning different component frame rates
  - Dynamic output dimension detection (440 vs previous 360)
- **Working End-to-End System**
  - Fixed VQ-VAE attention head dimension issues (changed from 6 to 8 heads)
  - Complete pipeline now functional with `test_embedding_validation.py`
  - Real-time audio processing with pretrained model accuracy
  - All 4 embedding components operational (Basic Pitch + Encodec + VQ-VAE + CLAP)

#### Changed - Documentation Consolidation
- **Major Cleanup**: Removed 6 outdated/redundant documentation files
  - Removed: NEXT_STEPS.md, QUICKSTART_EMBEDDING_VALIDATION.md (outdated)
  - Removed: documentation/TAB_GENERATION_PLAN.md (planning phase complete)
  - Removed: documentation/decoding_pondering_*.md (research notes)
  - Removed: documentation/README.md (unnecessary index)
- **Consolidated**: ARCHITECTURE_REFERENCE.md + IMPLEMENTATION_STATUS.md → STATUS.md
- **Updated**: documentation/DIMENSIONS.md for 440-dim Basic Pitch output
- **Added**: DEPRECATED.md to track superseded components
- **Result**: Clean, focused documentation structure with single authoritative status document

#### Purpose
This major update replaces the custom Basic Pitch reimplementation with the real Spotify model, providing immediate access to pre-trained accuracy and eliminating the need for training the first pipeline stage. Documentation cleanup ensures maintainable, non-redundant project structure.

### January 2025 - Week 3

#### Changed - Major Architecture Refactoring
- **VQ-VAE Architecture Correction**
  - Clarified that Kena AI IS a VQ-VAE system (not separate components)
  - Refactored `KenaVQVAE` to focus on embeddings + discrete tokens (removed direct transcription heads)
  - Updated forward() method to return only `z_q`, `indices`, and `commitment_loss`
  - Preserved multimodal compatibility for future video fusion
- **Embedding Pipeline Updates** (`models/audio_embedding_pipeline.py`)
  - Removed extraction of onset/frame predictions from VQ-VAE
  - Updated Stage 3 to output embeddings + discrete tokens only
  - Maintained 768-dimensional fused embeddings for multimodal fusion
- **Guitar Transcription System** (`models/guitar_transcription_system.py`)
  - Added `EmbeddingValidationDecoder` as audio_decoder for transcription
  - Updated transcribe() method to use embeddings → decoder pipeline
  - Separated embedding generation from transcription for multimodal compatibility
- **Documentation Consolidation**
  - Created `ARCHITECTURE_REFERENCE.md` as single source of architectural truth
  - Updated all mermaid diagrams to show audio decoder instead of direct VQ-VAE transcription
  - Refactored `DIMENSIONS.md` to reflect embeddings-first approach
  - Updated `IMPLEMENTATION_STATUS.md` with current architectural state

#### Purpose
This refactoring corrects a fundamental architectural misunderstanding where VQ-VAE was doing direct transcription, breaking the multimodal vision. The new architecture preserves 768-dim embeddings for future video fusion while providing audio transcription via separate decoders.

#### Added - Embedding Validation System
- **CRNN Validation Decoder** (`models/embedding_validation_decoder.py`)
  - CRNN architecture following GAPS paper research (CNN → GRU → Multi-head outputs)
  - Guitar-specific pitch bias towards E2-E6 range
  - Multi-task learning with onset + frame predictions
  - Frozen embedding pipeline testing to assess current quality
- **Complete Validation Pipeline** (`EmbeddingValidationSystem`)
  - Audio → Embeddings → CRNN Decoder → Tablature transcription
  - Quality metrics computation (sparsity, activation patterns, pitch coverage)
  - Component ablation capability for debugging individual embedding stages
- **Validation Test Script** (`test_embedding_validation.py`)
  - Synthetic guitar audio testing
  - Immediate embedding quality assessment
  - Debugging guidance for missing components
- **Documentation Updates**
  - Updated implementation status to reflect validation focus
  - Added quickstart guide for embedding validation
  - Clarified current priorities and next steps

#### Purpose
This addition enables immediate testing of the 4-stage embedding pipeline quality without requiring full system training. The CRNN decoder serves as a diagnostic tool to assess whether the Basic Pitch → Encodec → VQ-VAE → CLAP pipeline produces embeddings suitable for guitar transcription.

## [Previous Releases]

### December 2024

#### Added - Complete Transcription System
- **Kena-Style Pitch Decoder** (`models/pitch_decoder.py`)
  - Dual-loss architecture (onset + frame detection)
  - Multi-head outputs matching Kena AI's 87% accuracy approach
  - Guitar-specific pitch bias initialization
  - Confidence scoring for predictions
  - Frame-to-note post-processing with peak picking
- **Tab Assignment System** (`models/tab_assignment.py`)
  - Dynamic programming algorithm for optimal string/fret assignment
  - Physical constraint modeling (max 5-fret stretch)
  - Technique detection (slides, bends, hammer-ons, pull-offs)
  - ASCII tablature formatter
- **Complete Transcription System** (`models/guitar_transcription_system.py`)
  - End-to-end pipeline: Audio → Embeddings → Pitch → Notes → Tabs
  - Modular architecture matching contemporary systems
  - Training infrastructure with KenaStyleTrainer
  - File processing capabilities
  - Checkpoint saving/loading support
- **Usage Examples** (`example_usage.py`)
  - Complete demonstration of transcription pipeline
  - Component-by-component examples
  - Training setup guidance

#### Added - Audio Preprocessing
- Complete audio preprocessing pipeline implementation
  - `AudioProcessor` class in `utils/audio_processing.py`
  - Proper CQT computation with nnAudio GPU acceleration
  - Harmonic CQT stacking for guitar-specific pitch detection
  - Mel-spectrogram computation for CLAP encoder
  - Audio loading, resampling, and normalization utilities
  - Chunking support for long audio files
- Updated model components to use real audio features
  - `BasicPitchFeatureExtractor` now uses AudioProcessor for CQT
  - `CLAPAudioEncoder` now uses AudioProcessor for mel-spectrograms
- Testing infrastructure
  - `test_audio_preprocessing.py` with synthetic audio generation
  - Visualization capabilities for audio features
- Package structure improvements
  - Added `__init__.py` files for proper module imports
  - Created `requirements.txt` with all dependencies

#### Changed
- Replaced placeholder audio processing with real implementations
- Updated architecture to follow contemporary approaches (neural pitch + algorithmic tab)
- Reorganized documentation for clarity
- Updated all diagrams to reflect complete pipeline
- Main README now focuses on complete transcription system

#### Fixed
- Import paths between modules
- Dimension calculations for audio processing
- Architecture alignment with research best practices

### Initial Implementation (November 2024)

#### Added
- Core pipeline architecture
- Four-component hierarchical design:
  - Basic Pitch feature extractor
  - Meta Encodec wrapper
  - Guitar VQ-VAE
  - CLAP audio encoder
- Flexible dimension handling
- End-to-end pipeline flow
- Comprehensive documentation:
  - Project README
  - Implementation ROADMAP
  - Dimension guide
  - Research background

## Versioning

This project uses [Semantic Versioning](https://semver.org/). 

Current version: 0.1.0-alpha (Pre-release)