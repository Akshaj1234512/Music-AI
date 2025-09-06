# Fretting Transformer

A T5-based encoder-decoder model for automated MIDI-to-guitar tablature transcription, implementing the approach described in "Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription" by Hamberger et al.

## Overview

The Fretting Transformer treats guitar tablature generation as a text-to-text translation problem, converting MIDI sequences into guitar-specific notation that includes string and fret position information. The system addresses key challenges including string-fret ambiguity and physical playability through context-sensitive processing and post-processing validation.

### Key Features

- **T5 Encoder-Decoder Architecture**: Reduced T5 model (d_model=128, d_ff=1024, 3 layers, 4 heads)
- **Chunked Inference**: Processes long sequences in 20-note chunks with context preservation
- **Post-Processing**: Achieves 100% pitch accuracy through overlap correction and neighbor search
- **Comprehensive Evaluation**: Three metrics from paper - pitch accuracy, tab accuracy, and playability score
- **SynthTab Integration**: Direct support for SynthTab dataset with JAMS annotation format

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

**Input Tokens (MIDI Events):**
- `NOTE_ON<pitch>`: MIDI note start (pitch 0-127)
- `NOTE_OFF<pitch>`: MIDI note end (pitch 0-127)  
- `TIME_SHIFT<ticks>`: Time duration in MIDI ticks
- `CAPO<position>`: Capo position 0-7 (conditional)
- `TUNING<E,A,D,G,B,E>`: String tunings (conditional)

**Output Tokens (Tablature):**
- `TAB<string,fret>`: String (1-6) and fret (0-24) combination
- `TIME_SHIFT<ticks>`: Time duration matching input

### Evaluation Metrics (from Paper)

1. **Pitch Accuracy**: Percentage of correct pitches (allows alternative fingerings)
2. **Tab Accuracy**: Agreement with ground-truth string/fret combinations  
3. **Playability Score**: Difficulty based on finger stretches and movements

## Project Structure

```
fretting_transformer/
├── src/
│   ├── data/
│   │   ├── synthtab_loader.py      # SynthTab/JAMS data loading
│   │   ├── tokenizer.py            # MIDI/tablature tokenization
│   │   └── dataset.py              # PyTorch dataset and data processing
│   ├── model/
│   │   ├── config.py               # T5 model configuration
│   │   └── fretting_t5.py          # T5 model wrapper
│   ├── training/
│   │   ├── train.py                # Training pipeline with Adafactor
│   │   └── utils.py                # Training utilities and monitoring
│   ├── inference/
│   │   ├── generate.py             # Chunked inference system
│   │   └── postprocess.py          # Pitch validation and correction
│   └── evaluation/
│       └── metrics.py              # Paper evaluation metrics
├── scripts/
│   ├── prepare_data.py             # Data preparation script
│   ├── train_model.py              # Main training script
│   └── evaluate.py                 # Evaluation script
├── configs/
│   └── default_config.yaml         # Configuration template
├── JAMS_to_MIDI/                   # SynthTab utilities (existing)
└── experiments/                    # Output directory
```

## Paper Results Comparison

The paper reports the following results on test datasets:

| Dataset    | Tab Accuracy | Difficulty Score |
|-----------|-------------|------------------|
| GuitarToday| ~98%        | ~1.95           |
| Leduc      | ~72%        | ~4.24           |
| DadaGP     | ~82%        | ~2.41           |

Post-processing improves pitch accuracy from ~97% to 100% and tab accuracy from ~68% to 72%+.

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