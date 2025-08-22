# Basic Pitch Guitar Transcription Evaluation Pipeline

This project provides a complete pipeline for evaluating the performance of the Basic Pitch audio-to-MIDI transcription model, specifically tailored for the GuitarSet dataset. It handles everything from data loading and transcription to detailed, multi-faceted evaluation.

## Key Features

- **Automated Transcription**: Runs audio files through the Basic Pitch model to generate MIDI transcriptions.
- **Flexible Data Loading**: Intelligently pairs audio and ground truth MIDI files, with custom logic to match different naming conventions (e.g., `file.wav` and `file_mic.wav`).
- **Comprehensive Evaluation**: Calculates a suite of standard music information retrieval (MIR) metrics, including:
  - **Note-Level Metrics**: F1, Precision, and Recall for note accuracy (correct pitch and timing).
  - **Frame-Level Metrics**: F1, Precision, and Recall for frame-by-frame accuracy.
  - **Onset-Level Metrics**: F1, Precision, and Recall for the accuracy of note start times.
- **Configurable Pipeline**: All key parameters, including file paths, model thresholds, and output settings, are managed through a central `config.yaml` file.
- **Detailed Reporting**: Generates aggregate metrics with mean and standard deviation across all processed files, providing a robust overview of model performance.

## Setup

1.  **Clone the repository and install dependencies:**
    ```bash
    git clone <repository-url>
    cd MIDI_basic_pitch
    pip install -r requirements.txt
    ```

2.  **Configure the pipeline:**
    - Open `config.yaml`.
    - Set the `dataset_dir` to the root path of your GuitarSet dataset.
    - Ensure the `audio_dir_name` and `midi_dir_name` correspond to the correct subdirectories.

## Usage

To run the full evaluation on your dataset, execute the main script from the project's root directory:

```bash
python main.py --config config.yaml
```

### Limiting Files for Quick Tests

To run a quick test on a smaller subset of files, use the `--max-files` argument:

```bash
# Run on the first 5 files
python main.py --config config.yaml --max-files 5
```

## Configuration (`config.yaml`)

The `config.yaml` file is the control center for the pipeline. Here are the most important parameters:

- **`dataset_dir`**: The absolute path to your dataset (e.g., GuitarSet).
- **`output_dir`**: Where to save prediction outputs (generated MIDI, raw model probabilities).
- **`basic_pitch.onset_threshold`**: The confidence threshold (0.0 to 1.0) for detecting a note onset. **This is a critical parameter to tune.**
- **`basic_pitch.frame_threshold`**: The confidence threshold for detecting an active note in a given frame. **This is also critical to tune.**
- **`save_predictions`**: Set to `true` to save the transcribed MIDI files.
- **`save_model_outputs`**: Set to `true` to save the raw model probability matrices (`.npz` files).

## Core Modules

- **`main.py`**: The main entry point for the script. It orchestrates the data loading, transcription, and evaluation process.
- **`config.yaml`**: The configuration file for managing all pipeline parameters.
- **`basic_pitch_loader.py`**: Handles finding and pairing audio/MIDI files and loading ground truth data.
- **`basic_pitch_wrapper.py`**: A wrapper around the Basic Pitch library that manages the transcription process.
- **`evaluation_metrics.py`**: Contains all the logic for calculating note, frame, and onset evaluation metrics.


This directory contains tools for evaluating Spotify's Basic Pitch model on guitar transcription using the GuitarSet dataset. The pipeline runs end-to-end audio-to-MIDI transcription and computes detailed evaluation metrics.

## Overview

Basic Pitch is a lightweight, pre-trained model for automatic music transcription that converts audio to MIDI. This evaluation framework:

1. **Loads GuitarSet audio files** (.wav format)
2. **Runs Basic Pitch inference** to predict MIDI 
3. **Compares predictions against ground truth** MIDI annotations
4. **Computes comprehensive metrics** (frame F1, onset F1, note F1, etc.)

## Key Features

- **Pre-trained Model**: Uses Spotify's official Basic Pitch model (no training required)
- **TensorFlow + GPU**: Leverages TensorFlow backend with GPU acceleration
- **Comprehensive Metrics**: Frame-level, onset-level, and note-level evaluation
- **Raw Model Access**: Saves model probabilities for detailed analysis
- **GuitarSet Integration**: Designed specifically for guitar transcription evaluation

## Files

- `config.yaml` - Configuration for thresholds, paths, and GPU settings
- `basic_pitch_wrapper.py` - Main wrapper for Basic Pitch inference
- `basic_pitch_loader.py` - Data loader for audio and MIDI file pairs
- `evaluation_metrics.py` - Evaluation metrics (frame F1, onset F1, note F1)
- `main.py` - Main evaluation script
- `test_pipeline.py` - Test script to verify setup
- `README.md` - This documentation

## Installation Requirements

```bash
# Install Basic Pitch with TensorFlow backend
pip install basic-pitch[tf]

# Additional dependencies for evaluation
pip install mir_eval
pip install pyyaml
pip install pretty_midi
```

## Configuration

Edit `config.yaml` to set your paths and parameters:

```yaml
# Dataset paths (REQUIRED - user must fill these in)
paths:
  guitarset_audio_dir: "/path/to/GuitarSet/audio"     # GuitarSet .wav files
  guitarset_midi_dir: "/path/to/GuitarSet/annotation" # GuitarSet .mid files
  output_dir: "results"                               # Output directory

# Basic Pitch parameters (can be tuned)
basic_pitch:
  onset_threshold: 0.3      # Onset detection sensitivity
  frame_threshold: 0.3      # Frame activation threshold  
  minimum_frequency: 75.0   # Filter low frequencies (below guitar range)
  maximum_frequency: 2000.0 # Filter high frequencies (harmonics)
```

## Usage

### Single File Transcription

```python
from basic_pitch_wrapper import BasicPitchGuitarWrapper

# Initialize wrapper
wrapper = BasicPitchGuitarWrapper("config.yaml")

# Transcribe single file
result = wrapper.transcribe_single("guitar_audio.wav")

# Access outputs
model_output = result['model_output']    # Raw probabilities
midi_data = result['midi_data']          # PrettyMIDI object
note_events = result['note_events']      # List of (start, end, pitch, amplitude)
```

### Batch Processing

```python
# Process multiple files
audio_files = ["guitar1.wav", "guitar2.wav", "guitar3.wav"]
results = wrapper.transcribe_batch(audio_files)

# Results contains all outputs for each file
for audio_path, result in results.items():
    if 'error' not in result:
        print(f"{audio_path}: {len(result['note_events'])} notes detected")
```

## Outputs

For each processed audio file, the system generates:

### 1. Predicted MIDI File
- `{filename}_basic_pitch.mid` - Standard MIDI file with predicted notes

### 2. Raw Model Outputs  
- `{filename}_model_output.npz` - Numpy archive containing:
  - `note`: Frame-level note probabilities [time, 88_notes]
  - `onset`: Frame-level onset probabilities [time, 88_notes]  
  - `contour`: Pitch salience/contour information [time, 264_bins]

### 3. Note Events
- `{filename}_note_events.txt` - Text file with detected notes:
  ```
  # Format: start_time, end_time, midi_note, amplitude
  0.123456, 0.567890, 64, 0.789123
  ```

## Basic Pitch Model Details

**What Basic Pitch Returns:**
- `model_output`: Raw neural network probabilities
- `midi_data`: Processed MIDI file (PrettyMIDI format)  
- `note_events`: Note events as (start, end, pitch, amplitude, pitch_bends)

**Key Parameters:**
- `onset_threshold`: Minimum probability for note onset detection
- `frame_threshold`: Minimum probability for note frame activation
- `minimum_frequency`: Low-frequency cutoff (filters out non-guitar frequencies)
- `maximum_frequency`: High-frequency cutoff (focuses on guitar + harmonics)

## Evaluation Metrics (Coming Next)

The system will compute:

### Frame-Level Metrics
- `frame_f1`: F1 score for frame-wise note activation
- `frame_precision`: Precision of frame-wise predictions  
- `frame_recall`: Recall of frame-wise predictions

### Onset-Level Metrics  
- `onset_f1`: F1 score for note onset detection
- `onset_precision`: Precision of onset timing
- `onset_recall`: Recall of onset detection

### Note-Level Metrics (via mir_eval)
- `note_f1`: F1 score for complete note detection (onset + offset + pitch)
- `note_precision`: Precision of note detection
- `note_recall`: Recall of note detection

## GPU Requirements

- **TensorFlow GPU** support
- **CUDA-compatible GPU** recommended for fast inference
- **4GB+ GPU memory** for processing audio files
- **Automatic memory growth** configured to avoid allocation issues

## GuitarSet Dataset Structure

Expected directory structure:
```
GuitarSet/
├── audio/
│   ├── 00_BN1-129-Eb_comp.wav
│   ├── 00_BN2-119-D_comp.wav
│   └── ...
└── annotation/
    ├── 00_BN1-129-Eb_comp.mid
    ├── 00_BN2-119-D_comp.mid  
    └── ...
```

## Technical Notes

- **Sample Rate**: GuitarSet audio is 44.1kHz, Basic Pitch resamples to 22.05kHz internally
- **Backend**: Uses TensorFlow backend (recommended over ONNX/CoreML for this use case)
- **Memory**: Processes one file at a time to manage memory usage
- **Frequency Range**: Configured for guitar (E2=82Hz to ~2kHz including harmonics)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install basic-pitch[tf] mir_eval pyyaml pretty_midi librosa scikit-learn
   ```

2. **Set up your config.yaml:**
   ```yaml
   paths:
     guitarset_audio_dir: "/path/to/your/audio/files"
     guitarset_midi_dir: "/path/to/your/midi/files"
   ```

3. **Test the pipeline:**
   ```bash
   python test_pipeline.py
   ```

4. **Run evaluation on a few files first:**
   ```bash
   python main.py --config config.yaml --max-files 5
   ```

5. **Run on full dataset:**
   ```bash
   python main.py --config config.yaml --output results.json
   ```

## Expected Output Format

The evaluation will output aggregate metrics including:
- **Note F1, Precision, Recall** (using mir_eval)
- **Frame F1, Precision, Recall** (frame-level transcription)
- **Onset F1, Precision, Recall** (onset detection accuracy)

Results are displayed in console and optionally saved to JSON for detailed analysis.

## Command Line Options

- `--config`: Path to YAML config file (default: config.yaml)
- `--output`: Path to save JSON results file
- `--max-files`: Limit number of files (for testing)
- `--verbose`: Enable detailed logging