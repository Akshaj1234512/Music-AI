# Fretting Transformer

A T5-based encoder-decoder model for automated MIDI-to-guitar tablature transcription, implementing the approach described in "Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription" by Hamberger et al.

## Overview

The Fretting Transformer treats guitar tablature generation as a text-to-text translation problem, converting MIDI sequences into guitar-specific notation that includes string and fret position information. The system addresses key challenges including string-fret ambiguity and physical playability through context-sensitive processing and post-processing validation.

### Key Features

- **T5 Encoder-Decoder Architecture**: Reduced T5 model (d_model=128, d_ff=1024, 3 layers, 4 heads)
- **Chunked Inference**: Processes long sequences in 20-note chunks with context preservation
- **Post-Processing**: Pitch validation and correction through overlap correction and neighbor search
- **Comprehensive Evaluation**: Three metrics from paper - pitch accuracy, tab accuracy, and playability score
- **SynthTab Integration**: Direct support for SynthTab dataset with JAMS annotation format
- **GuitarSet Fine-tuning**: Professional guitar recordings for domain adaptation and evaluation

## Installation

### Requirements

```bash
torch >= 1.9.0
transformers >= 4.21.0
mido >= 1.2.0
numpy >= 1.20.0
matplotlib >= 3.3.0
scikit-learn >= 1.0.0
tqdm >= 4.60.0
pyyaml >= 6.0
```

### Setup

1. Clone the repository and navigate to the project directory
2. Install dependencies:
```bash
pip install torch transformers mido numpy matplotlib scikit-learn tqdm pyyaml
```

3. Ensure SynthTab dataset is accessible at `/data/andreaguz/SynthTab_Dev` or update paths in configuration

## Quick Start

### 🚀 **One-Command Pipeline (Recommended)**

Run the complete pipeline with a single command:

```bash
# Full paper reproduction
python run_pipeline.py --output_dir experiments/my_experiment

# Quick test with small dataset and debug model
python run_pipeline.py \
    --max_files 10 \
    --model_type debug \
    --num_epochs 5 \
    --output_dir experiments/test_run

# GPU-optimized training with mixed precision
python run_pipeline.py \
    --gpu_ids "0" \
    --use_fp16 \
    --batch_size 64 \
    --apply_postprocessing \
    --compare_baseline \
    --output_dir experiments/gpu_run
```

The pipeline automatically handles:
- ✅ Data preparation and caching
- ✅ Model training with checkpointing
- ✅ Final evaluation with metrics
- ✅ Organized output directory structure

### 🎛️ **Individual Stages (Advanced)**

For more control, run individual stages:

```bash
# Only data preparation
python run_pipeline.py --stage data --output_dir experiments/data_only

# Only training (assumes prepared data exists)
python run_pipeline.py --stage train --output_dir experiments/existing_data

# Only evaluation (assumes trained model exists)  
python run_pipeline.py --stage eval --output_dir experiments/existing_model
```

### 🔧 **Manual Step-by-Step (Legacy)**

<details>
<summary>Click to expand manual workflow</summary>

#### 1. Data Preparation
```bash
python scripts/prepare_data.py \
    --synthtab_path /data/andreaguz/SynthTab_Dev \
    --data_category jams \
    --max_files 50 \
    --analyze_data \
    --save_tokenizer
```

#### 2. Training
```bash
python scripts/train_model.py \
    --synthtab_path /data/andreaguz/SynthTab_Dev \
    --data_category jams \
    --max_files 100 \
    --model_type paper \
    --num_epochs 100 \
    --batch_size 32 \
    --output_dir experiments/checkpoints
```

#### 3. Evaluation
```bash
python scripts/evaluate.py \
    --model_path experiments/checkpoints/checkpoint-5000 \
    --synthtab_path /data/andreaguz/SynthTab_Dev \
    --apply_postprocessing \
    --compare_baseline \
    --output_dir experiments/evaluation
```

</details>

## GuitarSet Fine-tuning

### 🎸 **Professional Guitar Recording Dataset**

Fine-tune the model on GuitarSet - 360 professional guitar recordings (30 seconds each) with comprehensive tablature annotations. GuitarSet provides real-world performance data across multiple musical styles and playing techniques.

#### **Dataset Overview**
- **360 excerpts**: 6 players × 2 modes (comp/solo) × 5 styles × 3 progressions × 2 tempi
- **Musical styles**: Jazz, Rock, Bossa Nova, Singer-Songwriter, Funk
- **Ground truth**: Per-string MIDI annotations with precise timing
- **Total duration**: ~3 hours of professional guitar performances

#### **Quick Start: GuitarSet Fine-tuning**

```bash
# 1. Prepare GuitarSet data splits
python scripts/prepare_guitarset_data.py

# 2. Quick test (small subset)
python scripts/finetune_guitarset.py \
    --quick_test \
    --gpu_id 0 \
    --output_dir /data/andreaguz/guitarset_test

# 3. Full fine-tuning
python scripts/finetune_guitarset.py \
    --gpu_id 0 \
    --batch_size 16 \
    --num_epochs_stage1 5 \
    --num_epochs_stage2 15 \
    --use_fp16 \
    --output_dir /data/andreaguz/guitarset_experiment
```

#### **Two-Stage Fine-tuning Strategy**
1. **Stage 1**: Frozen encoder, fine-tune decoder only (5 epochs)
2. **Stage 2**: Full model fine-tuning with lower learning rate (15 epochs)

#### **Evaluate Trained Model**

Test on individual excerpts:
```bash
# Evaluate on Bossa Nova composition
python scripts/evaluate_sample.py "00_BN1-129-Eb_comp" \
    --model_path /data/andreaguz/guitarset_experiment/checkpoint-2000 \
    --gpu_id 0 \
    --output_dir /data/andreaguz/sample_results \
    --save_tokens

# Compare different styles
python scripts/evaluate_sample.py "01_Jazz1-130-D_solo" --model_path ...
python scripts/evaluate_sample.py "02_Rock1-130-A_comp" --model_path ...
```

#### **Results and Output Structure**
```
/data/andreaguz/guitarset_experiment/
├── checkpoint-1000/                    # Model checkpoints
│   ├── pytorch_model.bin              # Model weights
│   ├── config.json                    # Model configuration
│   └── tokenizer.json                 # Tokenizer
├── training_20231208_143022.log       # Training logs
├── training_config.json               # Experiment configuration
└── ...

/data/andreaguz/sample_results/
├── 00_BN1-129-Eb_comp_evaluation.json # Detailed metrics
└── 00_BN1-129-Eb_comp_summary.txt     # Human-readable results
```

#### **Key Features**
- **Player-based splits**: Avoid data leakage with proper train/val/test division
- **GPU support**: Flexible GPU selection and mixed precision training
- **Real-time monitoring**: Comprehensive logging and checkpoint management
- **Style diversity**: Train and evaluate across multiple musical genres
- **Professional data**: Bridge the gap between synthetic and real guitar performances

## Architecture Details

### Model Specifications (from Paper)

- **Architecture**: T5 Encoder-Decoder
- **Model Dimension**: 128
- **Feedforward Dimension**: 1024
- **Layers**: 3 encoder + 3 decoder
- **Attention Heads**: 4
- **Optimizer**: Adafactor with adaptive learning rate
- **Sequence Length**: 512 tokens
- **Inference Chunks**: 20 notes (~100 tokens)

### Tokenization

**🆕 Unified Vocabulary (Current Implementation):**

The system uses a **single unified vocabulary** (468 tokens) for both input and output, ensuring compatibility with standard T5 architecture:

**Special Tokens:**
- `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`: Standard sequence markers

**MIDI Input Events:**
- `NOTE_ON<pitch>`: MIDI note start (pitch 0-127) 
- `NOTE_OFF<pitch>`: MIDI note end (pitch 0-127)
- `TIME_SHIFT<ticks>`: Time duration in MIDI ticks
- `CAPO<position>`: Capo position 0-7 (optional)
- `TUNING<E,A,D,G,B,E>`: String tunings (optional)

**Tablature Output Events:**
- `TAB<string,fret>`: String (1-6) and fret (0-24) combination
- `TIME_SHIFT<ticks>`: Time duration matching input

**Key Architecture Fix**: Unlike previous implementations with separate encoder/decoder vocabularies that caused training failures, this unified approach uses a single vocabulary for both input and output sequences, fixing fundamental T5 compatibility issues.

### Evaluation Metrics (from Paper)

1. **Pitch Accuracy**: Percentage of correct pitches (allows alternative fingerings)
2. **Tab Accuracy**: Agreement with ground-truth string/fret combinations  
3. **Playability Score**: Difficulty based on finger stretches and movements

## Project Structure

```
fretting_transformer/
├── src/
│   ├── data/
│   │   ├── synthtab_loader.py           # SynthTab/JAMS data loading
│   │   ├── unified_tokenizer.py         # 🆕 Unified vocabulary tokenizer (468 tokens)
│   │   ├── unified_dataset.py           # 🆕 Unified dataset processor
│   │   ├── guitarset_loader.py          # 🎸 GuitarSet JAMS parser with tablature
│   │   ├── guitarset_dataset.py         # 🎸 PyTorch dataset for GuitarSet
│   │   ├── tokenizer.py                 # Legacy dual-vocab tokenizer
│   │   └── dataset.py                   # Legacy dataset processor
│   ├── model/
│   │   ├── config.py                    # T5 model configuration
│   │   ├── unified_fretting_t5.py       # 🆕 Standard T5 with unified vocab
│   │   └── fretting_t5.py               # Legacy model with custom heads
│   ├── training/
│   │   ├── train.py                     # Training pipeline with Adafactor
│   │   └── utils.py                     # Training utilities and monitoring
│   ├── inference/
│   │   ├── generate.py                  # Chunked inference system
│   │   └── postprocess.py               # Pitch validation and correction
│   └── evaluation/
│       └── metrics.py                   # Paper evaluation metrics
├── scripts/
│   ├── prepare_data.py                  # Legacy data preparation script
│   ├── train_model.py                   # Legacy training script  
│   ├── train_unified_model.py           # 🆕 Unified training script
│   ├── evaluate.py                      # Legacy evaluation script
│   ├── prepare_guitarset_data.py        # 🎸 GuitarSet data preparation
│   ├── finetune_guitarset.py           # 🎸 GuitarSet fine-tuning script
│   └── evaluate_sample.py               # 🎸 Single excerpt evaluation
├── run_pipeline.py                      # 🆕 Updated unified pipeline runner
├── test_unified_pipeline.py             # 🆕 Pipeline testing script
├── configs/
│   └── default_config.yaml              # Configuration template
├── JAMS_to_MIDI/                        # SynthTab utilities (existing)
└── experiments/                         # Output directory
```

**🆕 New Files (Unified Vocabulary):**
- `unified_tokenizer.py`: Single vocabulary for both MIDI and tablature
- `unified_dataset.py`: Dataset processor using unified approach  
- `unified_fretting_t5.py`: Standard T5 model (no custom heads)
- Updated `run_pipeline.py`: Uses unified components by default

**🎸 GuitarSet Integration:**
- `guitarset_loader.py`: Parse GuitarSet JAMS with per-string tablature extraction
- `guitarset_dataset.py`: PyTorch dataset for GuitarSet fine-tuning
- `prepare_guitarset_data.py`: Data splits and statistics generation
- `finetune_guitarset.py`: Two-stage fine-tuning pipeline
- `evaluate_sample.py`: Individual excerpt evaluation and analysis

**📦 Legacy Files**: Original dual-vocabulary files maintained for reference

## Implementation Status

### 🔧 **Current Architecture**

**Technical Implementation:**
- ✅ **T5 Compatibility**: Unified vocabulary approach eliminates encoder/decoder mismatch issues
- ✅ **Standard Architecture**: Uses HuggingFace T5ForConditionalGeneration without modifications
- ✅ **Stable Training**: No vocabulary errors or embedding table issues
- ✅ **GuitarSet Integration**: Complete pipeline for professional guitar recording fine-tuning

**Key Features:**
- **Unified Vocabulary**: Single vocabulary (468 tokens) for both MIDI input and tablature output
- **Two-Stage Fine-tuning**: Frozen encoder stage followed by full model training
- **Professional Data Support**: GuitarSet integration with 360 professionally recorded excerpts
- **Comprehensive Pipeline**: End-to-end training, evaluation, and sample testing

### 📊 **Paper Reference Results**

The original paper by Hamberger et al. reports results on various datasets:

| Dataset    | Tab Accuracy | Difficulty Score |
|-----------|-------------|------------------|
| GuitarToday| ~98%        | ~1.95           |
| Leduc      | ~72%        | ~4.24           |
| DadaGP     | ~82%        | ~2.41           |

Note: These are reference results from the original paper. Our implementation provides the framework to reproduce and extend these results.

## Configuration

The system uses YAML configuration files. Key parameters:

```yaml
# Model (Paper Specifications)
model:
  d_model: 128
  d_ff: 1024
  num_layers: 3
  num_heads: 4

# Training  
training:
  num_epochs: 100
  batch_size: 32
  optimizer: "adafactor"
  
# Inference
inference:
  chunk_size_notes: 20
  num_beams: 4
  apply_postprocessing: true
```

## GPU Configuration

The system automatically detects and uses GPU when available. Control GPU usage with these options:

### **GPU Selection**
```bash
# Use specific GPU
python run_pipeline.py --gpu_ids "1" --output_dir experiments/gpu1

# Use multiple GPUs  
python run_pipeline.py --gpu_ids "0,1" --output_dir experiments/multi_gpu

# Force CPU usage
python run_pipeline.py --force_cpu --output_dir experiments/cpu_only

# Mixed precision for faster training (requires compatible GPU)
python run_pipeline.py --use_fp16 --gpu_ids "0" --output_dir experiments/fp16
```

### **Memory Optimization**
```bash
# For limited GPU memory
python run_pipeline.py --batch_size 16 --model_type debug

# For high-memory GPUs
python run_pipeline.py --batch_size 64 --use_fp16
```

### **Environment Variables (Alternative)**
```bash
# Set before running
export CUDA_VISIBLE_DEVICES=1
python run_pipeline.py --output_dir experiments/gpu1

# Multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2
python run_pipeline.py --output_dir experiments/multi_gpu
```

## Output Directory Structure

The pipeline creates organized experiments:

```
experiments/my_experiment/
├── checkpoints/              # Model checkpoints
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── ...
├── logs/                     # Training logs and curves
│   ├── training.log
│   └── training_curves.png
├── data/                     # Processed data cache
│   ├── synthtab_cache.pkl
│   ├── tokenizer.json
│   └── dataset_info.json
├── evaluation/               # Evaluation results
│   ├── evaluation_metrics.json
│   └── evaluation_report.txt
└── pipeline_config.json      # Complete configuration
```

## Advanced Usage

### Custom Tokenization

```python
from src.data.tokenizer import FrettingTokenizer

tokenizer = FrettingTokenizer()
input_tokens, output_tokens = tokenizer.encode_sequence_pair(midi_events)
```

### Chunked Inference

```python
from src.inference.generate import ChunkedInference

inference = ChunkedInference(model, tokenizer)
input_tokens, output_tokens = inference.generate_tablature(midi_events)
```

### Post-Processing

```python
from src.inference.postprocess import apply_postprocessing

corrected_tokens, metrics = apply_postprocessing(
    input_tokens, output_tokens, tokenizer
)
print(f"Pitch accuracy after correction: {metrics['pitch_accuracy_after']:.2%}")
```

### Evaluation

```python
from src.evaluation.metrics import FrettingEvaluator

evaluator = FrettingEvaluator(tokenizer)
metrics = evaluator.evaluate_sequence(input_tokens, predicted_tokens, ground_truth_tokens)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Enable mixed precision with `--use_fp16`
3. **Poor Convergence**: Check learning rate and ensure data quality
4. **Low Tab Accuracy**: Enable post-processing and verify ground truth alignment

### 🆕 **Unified Vocabulary Issues**

5. **Import Errors**: Ensure you're using the updated `run_pipeline.py` that imports unified components
6. **Legacy Script Conflicts**: Old scripts (`train_model.py`, `evaluate.py`) use dual vocabularies and may fail
7. **Model Loading Issues**: Use models trained with unified approach; legacy models incompatible
8. **Generation Quality**: Model may need more training epochs or larger model size for better tablature output

### Performance Tips

- Use cached data preparation for faster iteration
- Enable mixed precision training for speed
- Use gradient accumulation for larger effective batch sizes
- Apply post-processing for production use

### Long-Running Training

For full dataset training that may take hours/days, use background execution:

```bash
# Run in background with nohup (continues after SSH disconnect)
nohup python run_pipeline.py \
    --output_dir /data/andreaguz/fretting_experiments/full_training \
    --synthtab_path /data/andreaguz/SynthTab_Dev \
    --gpu_ids "7" \
    --model_type paper \
    --num_epochs 100 \
    --batch_size 32 \
    --use_fp16 \
    --apply_postprocessing \
    --clean_start \
    > training_log.txt 2>&1 &

# Monitor progress
tail -f training_log.txt

# Check if still running
ps aux | grep run_pipeline
```

## Contributing

1. Follow the paper's specifications for model architecture
2. Maintain compatibility with SynthTab dataset format  
3. Add tests for new functionality
4. Update documentation for API changes

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{hamberger2024fretting,
  title={Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription},
  author={Hamberger, et al.},
  journal={...},
  year={2024}
}
```

## License

This implementation follows the original paper's approach and is intended for research and educational purposes.