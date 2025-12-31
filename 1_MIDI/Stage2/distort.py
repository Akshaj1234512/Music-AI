import os
import glob
import random
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve

# --- Configuration ---
guitarset_dir = '/data/akshaj/MusicAI/GuitarSet/audio_mix'
ir_dir = '/data/akshaj/MusicAI/irs'
output_dir = '/data/akshaj/MusicAI/GuitarSet/distorted'
TARGET_SR = 16000 
DISTORTION_GAIN = 15.0 

os.makedirs(output_dir, exist_ok=True)
random.seed(42)

def apply_clipping_distortion(signal, gain):
    return np.tanh(signal * gain)

guitar_files = sorted(glob.glob(os.path.join(guitarset_dir, '*.wav')))
ir_files = sorted(glob.glob(os.path.join(ir_dir, '*.wav')))

for g_file in guitar_files:
    try:
        # 1. Load clean guitar and IR at 16k
        signal, _ = librosa.load(g_file, sr=TARGET_SR, mono=True)
        ir_path = random.choice(ir_files)
        ir, _ = librosa.load(ir_path, sr=TARGET_SR, mono=True)

        # 2. Find the EXACT peak of the IR (The "Zero" point)
        # This accounts for even that 0.12ms mic-distance delay
        peak_idx = np.argmax(np.abs(ir))

        # 3. Apply Distortion
        clipped_signal = apply_clipping_distortion(signal, DISTORTION_GAIN)

        # 4. CONVOLUTION with TIMING CORRECTION
        # 'full' gives us the guitar length + IR length
        full_conv = fftconvolve(clipped_signal, ir, mode='full')

        # We slice starting at the PEAK of the IR and take the length of the signal
        # This aligns the Clean Pluck with the Distorted Pluck perfectly (0ms error)
        distorted_signal = full_conv[peak_idx : peak_idx + len(signal)]
        
        # 5. Peak Normalization
        max_val = np.max(np.abs(distorted_signal))
        if max_val > 0:
            distorted_signal = (distorted_signal / max_val) * 0.9
        
        # 6. Save
        out_name = os.path.basename(g_file)
        save_path = os.path.join(output_dir, out_name)
        sf.write(save_path, distorted_signal, TARGET_SR)

    except Exception as e:
        print(f"Error on {g_file}: {e}")

print("Done! Files regenerated with 0ms alignment error.")