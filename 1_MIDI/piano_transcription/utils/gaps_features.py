import numpy as np
import argparse
import os
import time
import logging
import h5py
import librosa
import csv
from utilities import (create_folder, float32_to_int16, create_logging,
    get_filename, normalize_audio)
import config


def read_gaps_midi(midi_path):
    """Parse MIDI file of GAPS dataset using pretty_midi.

    Args:
        midi_path: str, path to MIDI file

    Returns:
        dict: Dictionary containing MIDI events and timestamps in SECONDS
    """
    import pretty_midi

    midi_file = pretty_midi.PrettyMIDI(midi_path)

    midi_dict = {
        'midi_event': [],
        'midi_event_time': []
    }

    # Process ALL instruments and combine them into one channel
    for instrument in midi_file.instruments:
        for note in instrument.notes:
            # Note on event - combine all instruments into channel 0
            event_str = f"note_on channel=0 note={note.pitch} velocity={note.velocity} time={note.start}"
            midi_dict['midi_event'].append(event_str)
            midi_dict['midi_event_time'].append(note.start)

            # Note off event - combine all instruments into channel 0
            event_str = f"note_off channel=0 note={note.pitch} velocity=0 time={note.end}"
            midi_dict['midi_event'].append(event_str)
            midi_dict['midi_event_time'].append(note.end)

    # Sort events by time to ensure chronological order
    sorted_indices = np.argsort(midi_dict['midi_event_time'])
    midi_dict['midi_event'] = [midi_dict['midi_event'][i] for i in sorted_indices]
    midi_dict['midi_event_time'] = [midi_dict['midi_event_time'][i] for i in sorted_indices]

    return midi_dict


def load_splits(dataset_dir):
    """Load train/val/test splits from CSV.

    Args:
        dataset_dir: str, path to GAPS dataset

    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    splits_path = os.path.join(dataset_dir, 'gaps_v1_splits.csv')
    splits = {'train': [], 'val': [], 'test': []}

    with open(splits_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row['split']
            filename = row['filename'].replace('.wav', '')  # Remove extension
            if split in splits:
                splits[split].append(filename)

    return splits


def pack_gaps_dataset_to_hdf5(args):
    """Load & resample GAPS audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of GAPS dataset (containing midi/, audio/, etc.)
      workspace: str, directory of your workspace
      audio_dir: str, directory containing downloaded WAV files
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    audio_dir = args.audio_dir

    sample_rate = config.sample_rate

    # Paths
    midi_dir = os.path.join(dataset_dir, 'midi')
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'gaps')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Load splits
    splits_dict = load_splits(dataset_dir)
    logging.info(f"Train: {len(splits_dict['train'])}, Val: {len(splits_dict['val'])}, Test: {len(splits_dict['test'])}")

    # Find all MIDI files
    midi_files = [f for f in os.listdir(midi_dir) if f.endswith('-fine-aligned.mid')]
    midi_files.sort()
    logging.info(f'Total MIDI files: {len(midi_files)}')

    # Process each split
    for split_name in ['train', 'val', 'test']:
        split_filenames = set(splits_dict[split_name])

        # Create HDF5 directory for this split
        split_hdf5_dir = os.path.join(waveform_hdf5s_dir, '2024', split_name)
        create_folder(split_hdf5_dir)

        logging.info(f'\n=== Processing {split_name} split ===')

        count = 0
        for midi_file in midi_files:
            # Extract base name (e.g., "mvswc" from "mvswc-fine-aligned.mid")
            base_name = midi_file.replace('-fine-aligned.mid', '')

            # Check if this file belongs to current split
            if base_name not in split_filenames:
                continue

            midi_path = os.path.join(midi_dir, midi_file)
            audio_path = os.path.join(audio_dir, f'{base_name}.wav')

            # Check if audio exists
            if not os.path.exists(audio_path):
                logging.warning(f'Audio not found: {audio_path}, skipping')
                continue

            # Output HDF5 path
            hdf5_path = os.path.join(split_hdf5_dir, f'{base_name}.h5')

            # Skip if already processed
            if os.path.exists(hdf5_path):
                logging.info(f'{count+1}/{len(split_filenames)} Already exists: {hdf5_path}')
                count += 1
                continue

            logging.info(f'{count+1}/{len(split_filenames)} Processing: {base_name}')

            try:
                # Load audio
                (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                audio = normalize_audio(audio)

                # Parse MIDI
                midi_dict = read_gaps_midi(midi_path)

                # Write to HDF5
                with h5py.File(hdf5_path, 'w') as hf:
                    hf.attrs.create('split', data=split_name.encode(), dtype='S20')
                    hf.attrs.create('dataset', data='gaps'.encode(), dtype='S20')

                    # Create datasets
                    hf.create_dataset(name='midi_event',
                        data=[s.encode() for s in midi_dict['midi_event']],
                        dtype='S200')

                    hf.create_dataset(name='midi_event_time',
                        data=midi_dict['midi_event_time'],
                        dtype=np.float32)

                    hf.create_dataset(name='waveform',
                        data=float32_to_int16(audio),
                        dtype=np.int16)

                logging.info(f'  Saved: {hdf5_path}')
                logging.info(f'  Audio duration: {len(audio)/sample_rate:.2f}s, MIDI events: {len(midi_dict["midi_event"])}')

                count += 1

            except Exception as e:
                logging.error(f'Error processing {base_name}: {e}')
                continue

        logging.info(f'{split_name} split: Processed {count}/{len(split_filenames)} files')

    logging.info('\n=== GAPS dataset packing complete ===')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack GAPS dataset to HDF5 format')
    parser.add_argument('--dataset_dir', type=str, required=True,
        help='Directory of GAPS dataset (containing midi/, gaps_v1_splits.csv, etc.)')
    parser.add_argument('--workspace', type=str, required=True,
        help='Directory of your workspace')
    parser.add_argument('--audio_dir', type=str, required=True,
        help='Directory containing downloaded WAV audio files')

    args = parser.parse_args()
    pack_gaps_dataset_to_hdf5(args)
