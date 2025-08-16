#!/usr/bin/env python3
"""
Fix HDF5 Split Tags Script

This script fixes the split tags in GuitarSet HDF5 files to ensure proper train/validation split
without data leakage. It sets the first 80% of files to 'train' and last 20% to 'validation'.

Usage:
    python3 fix_hdf5_splits.py --hdf5_dir /path/to/hdf5s/guitarset/combined
"""

import os
import h5py
import argparse
import glob
from tqdm import tqdm

def fix_hdf5_splits(hdf5_dir):
    """Fix split tags in HDF5 files to prevent data leakage."""
    
    # Find all HDF5 files
    hdf5_files = glob.glob(os.path.join(hdf5_dir, '**/*.h5'), recursive=True)
    hdf5_files = sorted(hdf5_files)  
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Calculate split points
    total_files = len(hdf5_files)
    train_end = int(total_files * 0.8)
    
    print(f"Setting first {train_end} files to 'train'")
    print(f"Setting last {total_files - train_end} files to 'validation'")
    
    # Process files
    for i, hdf5_path in enumerate(tqdm(hdf5_files, desc="Fixing split tags")):
        try:
            with h5py.File(hdf5_path, 'r+') as hf:
                # Determine split based on position
                if i < train_end:
                    new_split = 'train'
                else:
                    new_split = 'validation'
                
                # Update the split attribute
                if 'split' in hf.attrs:
                    old_split = hf.attrs['split']
                    if isinstance(old_split, bytes):
                        old_split = old_split.decode()
                    hf.attrs['split'] = new_split.encode()
                    print(f"  {os.path.basename(hdf5_path)}: {old_split} → {new_split}")
                else:
                    hf.attrs['split'] = new_split.encode()
                    print(f"  {os.path.basename(hdf5_path)}: (no split) → {new_split}")
                    
        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
    
    print(f"\nSplit tags fixed successfully!")
    print(f"Training files: {train_end}")
    print(f"Validation files: {total_files - train_end}")

def main():
    parser = argparse.ArgumentParser(description='Fix HDF5 split tags to prevent data leakage')
    parser.add_argument('--hdf5_dir', type=str, required=True,
                       help='Directory containing HDF5 files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_dir):
        print(f"Error: Directory {args.hdf5_dir} does not exist")
        return
    
    fix_hdf5_splits(args.hdf5_dir)

if __name__ == '__main__':
    main()
