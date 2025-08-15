import numpy as np
import h5py
import logging
import torch
import torch.utils.data
from utilities import traverse_folder
from data_generator import Augmentor, collate_fn
import config
import librosa


class GuitarSetDataset(torch.utils.data.Dataset):
    """GuitarSet dataset for training."""
    
    def __init__(self, hdf5s_dir, segment_seconds, frames_per_second, 
            max_note_shift, augmentor=None):
        """Initialize GuitarSet dataset.
        
        Args:
          hdf5s_dir: str, directory containing HDF5 files
          segment_seconds: float, duration of each segment
          frames_per_second: int, frames per second
          max_note_shift: int, maximum note shift for augmentation
          augmentor: Augmentor object or None
        """
        
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.max_note_shift = max_note_shift
        self.augmentor = augmentor
        
        self.sample_rate = config.sample_rate
        self.segment_samples = int(segment_seconds * self.sample_rate)
        self.classes_num = config.classes_num
        
        # Find all HDF5 files
        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.hdf5_paths = hdf5_paths
        
        logging.info('Found {} HDF5 files in {}'.format(len(self.hdf5_paths), hdf5s_dir))
    
    def __len__(self):
        return len(self.hdf5_paths)
    
    def __getitem__(self, idx):
        """Get a segment of audio and MIDI data."""
        
        hdf5_path = self.hdf5_paths[idx]
        
        with h5py.File(hdf5_path, 'r') as hf:
            # Load waveform
            waveform = hf['waveform'][:]
            waveform = waveform.astype(np.float32)
            
            # Load MIDI events
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            
            # Get duration
            duration = hf.attrs['duration']
            
            # Random start time for segment
            if duration > self.segment_seconds:
                start_time = np.random.uniform(0, duration - self.segment_seconds)
            else:
                start_time = 0
            
            # Extract segment
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + self.segment_samples
            
            if end_sample > len(waveform):
                # Pad if needed
                waveform = np.pad(waveform, (0, end_sample - len(waveform)), 'constant')
            
            waveform = waveform[start_sample:end_sample]
            
            # Apply augmentation if specified
            if self.augmentor is not None:
                waveform = self.augmentor.augment(waveform)
            
            # Apply pitch shift if specified
            if self.max_note_shift > 0:
                note_shift = np.random.randint(-self.max_note_shift, self.max_note_shift + 1)
                if note_shift != 0:
                    waveform = librosa.effects.pitch_shift(waveform, sr=self.sample_rate, 
                        steps=note_shift, bins_per_octave=12)
            
            # Create data dictionary
            data_dict = {
                'waveform': waveform,
                'midi_events': midi_events,
                'midi_events_time': midi_events_time,
                'start_time': start_time,
                'audio_type': hf.attrs['audio_type'].decode() if 'audio_type' in hf.attrs else 'unknown'
            }
            
            return data_dict


class GuitarSetSampler:
    """Sampler for GuitarSet dataset."""
    
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, 
            batch_size, mini_data=False, random_seed=1234):
        """Initialize GuitarSet sampler.
        
        Args:
          hdf5s_dir: str, directory containing HDF5 files
          split: str, 'train' or 'validation'
          segment_seconds: float, duration of each segment
          hop_seconds: float, hop between segments
          batch_size: int, batch size
          mini_data: bool, use small subset for debugging
          random_seed: int, random seed
        """
        
        self.hdf5s_dir = hdf5s_dir
        self.split = split
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.batch_size = batch_size
        self.mini_data = mini_data
        self.random_state = np.random.RandomState(random_seed)
        
        # Find all HDF5 files
        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.segment_list = []
        
        n = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split:
                    audio_name = hdf5_path.split('/')[-1]
                    year = hf.attrs['year'].decode()
                    duration = hf.attrs['duration']
                    
                    start_time = 0
                    while (start_time + self.segment_seconds < duration):
                        self.segment_list.append([year, audio_name, start_time])
                        start_time += self.hop_seconds
                    
                    n += 1
                    if self.mini_data and n == 10:
                        break
        
        logging.info('{} segments: {}'.format(split, len(self.segment_list)))
        
        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)
    
    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1
                
                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)
                
                batch_segment_list.append(self.segment_list[index])
                i += 1
            
            yield batch_segment_list
    
    def __len__(self):
        return -1
    
    def state_dict(self):
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
    
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']
