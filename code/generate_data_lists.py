#!/usr/bin/env python3
"""
Generate train/val file lists for mintpain dataset.
Converts the MATLAB data preparation script to Python.

Output files:
- train_rgb.txt, train_depth.txt, train_thermal.txt
- val_rgb.txt, val_depth.txt, val_thermal.txt

Each line format: <file_path> <label> <video_id>
"""

import os
from pathlib import Path


def generate_data_lists(
    data_root: str,
    output_dir: str,
    train_sweeps_per_trial: int = 72,
):
    """
    Generate train/val file lists for mintpain dataset.
    
    Args:
        data_root: Root path to mintpain data
        output_dir: Directory to save output files
        train_sweeps_per_trial: Number of sweeps per trial for training (rest for validation)
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open all output files
    files = {
        'train_rgb': open(output_dir / 'train_rgb.txt', 'w'),
        'train_depth': open(output_dir / 'train_depth.txt', 'w'),
        'train_thermal': open(output_dir / 'train_thermal.txt', 'w'),
        'val_rgb': open(output_dir / 'val_rgb.txt', 'w'),
        'val_depth': open(output_dir / 'val_depth.txt', 'w'),
        'val_thermal': open(output_dir / 'val_thermal.txt', 'w'),
    }
    
    train_video_count = 1
    val_video_count = 1
    
    # Get all subject folders (sorted)
    subject_folders = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    for sub_id, subject_folder in enumerate(subject_folders, 1):
        print(f"Processing subject {sub_id}: {subject_folder.name}")
        
        # Get all trial folders
        trial_folders = sorted([d for d in subject_folder.iterdir() if d.is_dir()])
        
        for trial_id, trial_folder in enumerate(trial_folders, 1):
            print(f"  Trial {trial_id}: {trial_folder.name}")
            
            # Get all sweep folders
            sweep_folders = sorted([d for d in trial_folder.iterdir() if d.is_dir()])
            
            for sweep_id, sweep_folder in enumerate(sweep_folders, 1):
                # Extract label from folder name (e.g., "Sub01_Trial01_Sweep01_Label0" -> label=0)
                folder_name = sweep_folder.name
                label = None
                for part in folder_name.split('_'):
                    if part.startswith('Label'):
                        label = int(part.replace('Label', ''))
                        break
                
                if label is None:
                    print(f"    Warning: Could not extract label from {folder_name}, skipping")
                    continue
                
                # Determine if train or validation
                is_train = sweep_id <= train_sweeps_per_trial
                
                # Process RGB files
                rgb_dir = sweep_folder / 'RGB'
                if rgb_dir.exists():
                    rgb_files = sorted(rgb_dir.glob('*.jpg')) + sorted(rgb_dir.glob('*.png'))
                    for rgb_file in rgb_files:
                        line = f"{rgb_file} {label} {train_video_count if is_train else val_video_count}\n"
                        if is_train:
                            files['train_rgb'].write(line)
                        else:
                            files['val_rgb'].write(line)
                
                # Process Depth files
                depth_dir = sweep_folder / 'D'
                if depth_dir.exists():
                    depth_files = sorted(depth_dir.glob('*.png')) + sorted(depth_dir.glob('*.jpg'))
                    for depth_file in depth_files:
                        line = f"{depth_file} {label} {train_video_count if is_train else val_video_count}\n"
                        if is_train:
                            files['train_depth'].write(line)
                        else:
                            files['val_depth'].write(line)
                
                # Process Thermal files
                thermal_dir = sweep_folder / 'T'
                if thermal_dir.exists():
                    thermal_files = sorted(thermal_dir.glob('*.png')) + sorted(thermal_dir.glob('*.jpg'))
                    for thermal_file in thermal_files:
                        line = f"{thermal_file} {label} {train_video_count if is_train else val_video_count}\n"
                        if is_train:
                            files['train_thermal'].write(line)
                        else:
                            files['val_thermal'].write(line)
                
                # Increment video count
                if is_train:
                    train_video_count += 1
                else:
                    val_video_count += 1
    
    # Close all files
    for f in files.values():
        f.close()
    
    print(f"\nComplete! Generated files in {output_dir}")
    print(f"  Train videos: {train_video_count - 1}")
    print(f"  Val videos: {val_video_count - 1}")
    
    # Print file statistics
    for name, filepath in [
        ('train_rgb', output_dir / 'train_rgb.txt'),
        ('train_depth', output_dir / 'train_depth.txt'),
        ('train_thermal', output_dir / 'train_thermal.txt'),
        ('val_rgb', output_dir / 'val_rgb.txt'),
        ('val_depth', output_dir / 'val_depth.txt'),
        ('val_thermal', output_dir / 'val_thermal.txt'),
    ]:
        with open(filepath, 'r') as f:
            count = sum(1 for _ in f)
        print(f"  {name}: {count} samples")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate train/val file lists for mintpain dataset')
    parser.add_argument('--data_root', type=str, 
                        default='/home/gm/Workspace/ai-projects/pain_detection/data/mintpain',
                        help='Root path to mintpain data')
    parser.add_argument('--output_dir', type=str,
                        default='/home/gm/Workspace/ai-projects/pain_detection/data/mintpain',
                        help='Directory to save output files')
    parser.add_argument('--train_sweeps', type=int, default=72,
                        help='Number of sweeps per trial for training (rest for validation)')
    
    args = parser.parse_args()
    
    generate_data_lists(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_sweeps_per_trial=args.train_sweeps,
    )