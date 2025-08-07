from .audio_embedding_pipeline import GuitarAudioEmbeddingPipeline
from .clean_basic_pitch_wrapper import CleanBasicPitchWrapper
from .huggingface_clap import HuggingFaceCLAP
from .huggingface_encodec import HuggingFaceEncodec
from .vq_vae import KenaVQVAE, KenaDualLoss, GuitarVQVAE, GuitarVQVAELoss  # Backwards compatibility aliases
from .pitch_decoder import FrameToNotePP  # Still needed for post-processing
from .tab_assignment import (
    DynamicProgrammingTabAssignment,
    TechniqueDetector,
    TabFormatter,
    GuitarNote
)
from .guitar_transcription_system import (
    GuitarTranscriptionSystem,
    AudioTranscriptionTrainer,
    KenaVQVAETrainer,  # Backwards compatibility alias
    KenaStyleTrainer   # Backwards compatibility alias
)
from .embedding_validation_decoder import (
    EmbeddingValidationDecoder,
    EmbeddingValidationSystem,
    EmbeddingValidationTrainer
)

__all__ = [
    'GuitarAudioEmbeddingPipeline',
    'CleanBasicPitchWrapper',
    'HuggingFaceCLAP', 
    'HuggingFaceEncodec',
    'KenaVQVAE',
    'KenaDualLoss', 
    'GuitarVQVAE',  # Backwards compatibility
    'GuitarVQVAELoss',  # Backwards compatibility
    'FrameToNotePP',
    'DynamicProgrammingTabAssignment',
    'TechniqueDetector',
    'TabFormatter',
    'GuitarNote',
    'GuitarTranscriptionSystem',
    'AudioTranscriptionTrainer',
    'KenaVQVAETrainer',  # Backwards compatibility  
    'KenaStyleTrainer',  # Backwards compatibility
    'EmbeddingValidationDecoder',
    'EmbeddingValidationSystem',
    'EmbeddingValidationTrainer'
]