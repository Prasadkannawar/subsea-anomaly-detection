"""
NauticAI Dataset Preparation Script
====================================

This script prepares the SUIM (Semantic Underwater Image Segmentation) dataset
for YOLOv8 object detection training.

Features:
- Splits dataset into 80% train / 20% validation
- Moves images and corresponding labels
- Validates YOLO format annotations
- Provides dataset statistics
- Supports data augmentation configuration

Usage:
------
    # Basic usage (split existing dataset)
    python dataset_prepare.py --source_images ./raw_data/images --source_labels ./raw_data/labels
    
    # With custom split ratio
    python dataset_prepare.py --source_images ./raw_data/images --source_labels ./raw_data/labels --split_ratio 0.85
    
    # Validate existing dataset
    python dataset_prepare.py --validate_only --dataset_path ./dataset

Requirements:
-------------
    pip install opencv-python numpy pillow pyyaml

Author: NauticAI Team
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
import numpy as np


class DatasetPreparer:
    """
    Handles dataset preparation for YOLOv8 training.
    """
    
    def __init__(self, dataset_root: str = "./dataset"):
        """
        Initialize dataset preparer.
        
        Args:
            dataset_root: Root directory for the prepared dataset
        """
        self.dataset_root = Path(dataset_root)
        self.train_images = self.dataset_root / "images" / "train"
        self.val_images = self.dataset_root / "images" / "val"
        self.train_labels = self.dataset_root / "labels" / "train"
        self.val_labels = self.dataset_root / "labels" / "val"
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create dataset directory structure."""
        for directory in [self.train_images, self.val_images, 
                         self.train_labels, self.val_labels]:
            directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Dataset directories created at: {self.dataset_root}")
    
    def split_dataset(self, 
                     source_images: str,
                     source_labels: str,
                     split_ratio: float = 0.8,
                     seed: int = 42) -> Dict:
        """
        Split dataset into train and validation sets.
        
        Args:
            source_images: Path to source images directory
            source_labels: Path to source labels directory
            split_ratio: Train/val split ratio (default: 0.8 = 80% train)
            seed: Random seed for reproducibility
            
        Returns:
            dict: Statistics about the split
        """
        random.seed(seed)
        
        source_images_path = Path(source_images)
        source_labels_path = Path(source_labels)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in source_images_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        # Shuffle dataset
        random.shuffle(image_files)
        
        # Calculate split index
        split_idx = int(len(image_files) * split_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Copy files
        train_copied = self._copy_files(train_files, source_labels_path, 
                                       self.train_images, self.train_labels)
        val_copied = self._copy_files(val_files, source_labels_path,
                                     self.val_images, self.val_labels)
        
        stats = {
            'total_images': len(image_files),
            'train_images': len(train_files),
            'val_images': len(val_files),
            'train_copied': train_copied,
            'val_copied': val_copied,
            'split_ratio': split_ratio
        }
        
        self._print_split_summary(stats)
        return stats
    
    def _copy_files(self, image_files: List[Path], 
                   source_labels_path: Path,
                   dest_images: Path,
                   dest_labels: Path) -> int:
        """
        Copy image and label files to destination.
        
        Args:
            image_files: List of image file paths
            source_labels_path: Source labels directory
            dest_images: Destination images directory
            dest_labels: Destination labels directory
            
        Returns:
            int: Number of successfully copied pairs
        """
        copied_count = 0
        
        for img_file in image_files:
            # Copy image
            dest_img = dest_images / img_file.name
            shutil.copy2(img_file, dest_img)
            
            # Copy corresponding label file
            label_file = source_labels_path / f"{img_file.stem}.txt"
            
            if label_file.exists():
                dest_label = dest_labels / f"{img_file.stem}.txt"
                shutil.copy2(label_file, dest_label)
                copied_count += 1
            else:
                print(f"⚠ Warning: No label file found for {img_file.name}")
        
        return copied_count
    
    def _print_split_summary(self, stats: Dict):
        """Print dataset split summary."""
        print("\n" + "="*60)
        print("📊 DATASET SPLIT SUMMARY")
        print("="*60)
        print(f"Total Images:       {stats['total_images']}")
        print(f"Training Set:       {stats['train_images']} ({stats['split_ratio']*100:.0f}%)")
        print(f"Validation Set:     {stats['val_images']} ({(1-stats['split_ratio'])*100:.0f}%)")
        print(f"Train Pairs Copied: {stats['train_copied']}")
        print(f"Val Pairs Copied:   {stats['val_copied']}")
        print("="*60 + "\n")
    
    def validate_dataset(self) -> Dict:
        """
        Validate the dataset structure and annotations.
        
        Returns:
            dict: Validation statistics and issues
        """
        print("\n🔍 Validating dataset...")
        
        issues = {
            'missing_labels': [],
            'missing_images': [],
            'invalid_boxes': [],
            'empty_labels': []
        }
        
        stats = {
            'train': {'images': 0, 'labels': 0, 'boxes': 0, 'class_counts': {}},
            'val': {'images': 0, 'labels': 0, 'boxes': 0, 'class_counts': {}}
        }
        
        # Validate train set
        print("  Checking training set...")
        self._validate_split(self.train_images, self.train_labels, 
                           stats['train'], issues)
        
        # Validate validation set
        print("  Checking validation set...")
        self._validate_split(self.val_images, self.val_labels,
                           stats['val'], issues)
        
        self._print_validation_summary(stats, issues)
        return {'stats': stats, 'issues': issues}
    
    def _validate_split(self, images_dir: Path, labels_dir: Path,
                       stats: Dict, issues: Dict):
        """Validate a single split (train or val)."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        stats['images'] = len(image_files)
        
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            # Check if label exists
            if not label_file.exists():
                issues['missing_labels'].append(str(img_file))
                continue
            
            stats['labels'] += 1
            
            # Validate label content
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    issues['empty_labels'].append(str(label_file))
                    continue
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        issues['invalid_boxes'].append(
                            f"{label_file}:L{line_num} - Expected 5 values, got {len(parts)}"
                        )
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        # Validate normalization (values should be 0-1)
                        if not all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
                            issues['invalid_boxes'].append(
                                f"{label_file}:L{line_num} - Box values not normalized (0-1)"
                            )
                            continue
                        
                        # Count classes
                        stats['class_counts'][class_id] = stats['class_counts'].get(class_id, 0) + 1
                        stats['boxes'] += 1
                        
                    except ValueError as e:
                        issues['invalid_boxes'].append(
                            f"{label_file}:L{line_num} - Parse error: {e}"
                        )
                        
            except Exception as e:
                issues['invalid_boxes'].append(f"{label_file} - Read error: {e}")
    
    def _print_validation_summary(self, stats: Dict, issues: Dict):
        """Print validation summary."""
        print("\n" + "="*60)
        print("✅ DATASET VALIDATION SUMMARY")
        print("="*60)
        
        # Training set
        print(f"\n📁 Training Set:")
        print(f"  Images:       {stats['train']['images']}")
        print(f"  Labels:       {stats['train']['labels']}")
        print(f"  Total Boxes:  {stats['train']['boxes']}")
        print(f"  Class Distribution:")
        for class_id, count in sorted(stats['train']['class_counts'].items()):
            print(f"    Class {class_id}: {count} boxes")
        
        # Validation set
        print(f"\n📁 Validation Set:")
        print(f"  Images:       {stats['val']['images']}")
        print(f"  Labels:       {stats['val']['labels']}")
        print(f"  Total Boxes:  {stats['val']['boxes']}")
        print(f"  Class Distribution:")
        for class_id, count in sorted(stats['val']['class_counts'].items()):
            print(f"    Class {class_id}: {count} boxes")
        
        # Issues
        total_issues = sum(len(v) for v in issues.values())
        print(f"\n⚠️  Issues Found: {total_issues}")
        
        if issues['missing_labels']:
            print(f"  Missing Labels:  {len(issues['missing_labels'])}")
        if issues['empty_labels']:
            print(f"  Empty Labels:    {len(issues['empty_labels'])}")
        if issues['invalid_boxes']:
            print(f"  Invalid Boxes:   {len(issues['invalid_boxes'])}")
        
        if total_issues == 0:
            print("  ✓ No issues found! Dataset is ready for training.")
        else:
            print("\n  Run with --show_issues to see detailed issue list")
        
        print("="*60 + "\n")


class AugmentationConfig:
    """
    Data augmentation configuration for YOLOv8 training.
    These parameters will be used during training.
    """
    
    @staticmethod
    def get_augmentation_yaml() -> str:
        """
        Get augmentation configuration in YAML format.
        
        Returns:
            str: YAML configuration for augmentations
        """
        config = """
# Data Augmentation Configuration for Underwater Detection
# Add these parameters when training YOLOv8

# Geometric Augmentations
hsv_h: 0.015      # HSV-Hue augmentation (fraction)
hsv_s: 0.7        # HSV-Saturation augmentation (fraction)
hsv_v: 0.4        # HSV-Value augmentation (fraction)
degrees: 10.0     # Image rotation (+/- deg)
translate: 0.1    # Image translation (+/- fraction)
scale: 0.5        # Image scale (+/- gain)
shear: 0.0        # Image shear (+/- deg)
perspective: 0.0  # Image perspective (+/- fraction), range 0-0.001
flipud: 0.0       # Flip up-down probability
fliplr: 0.5       # Flip left-right probability (50% for underwater symmetry)

# Color Augmentations (simulating underwater conditions)
mosaic: 1.0       # Mosaic augmentation probability
mixup: 0.1        # MixUp augmentation probability
copy_paste: 0.0   # Copy-paste augmentation probability

# Blur (simulating water turbidity)
blur: 0.0         # Apply blur with kernel size = blur (use carefully)

# Training Command Example:
# yolo detect train data=data.yaml model=yolov8s.pt epochs=150 imgsz=640 \\
#   hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 mosaic=1.0 mixup=0.1
"""
        return config
    
    @staticmethod
    def save_augmentation_config(output_path: str = "augmentation_config.yaml"):
        """Save augmentation configuration to file."""
        with open(output_path, 'w') as f:
            f.write(AugmentationConfig.get_augmentation_yaml())
        print(f"✓ Augmentation config saved to: {output_path}")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Prepare SUIM dataset for YOLOv8 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split dataset (80/20)
  python dataset_prepare.py --source_images ./raw_data/images --source_labels ./raw_data/labels
  
  # Custom split ratio (85/15)
  python dataset_prepare.py --source_images ./raw_data/images --source_labels ./raw_data/labels --split_ratio 0.85
  
  # Validate existing dataset
  python dataset_prepare.py --validate_only
  
  # Generate augmentation config
  python dataset_prepare.py --augmentation_config_only
        """
    )
    
    parser.add_argument('--source_images', type=str, 
                       help='Path to source images directory')
    parser.add_argument('--source_labels', type=str,
                       help='Path to source labels directory')
    parser.add_argument('--dataset_root', type=str, default='./dataset',
                       help='Root directory for prepared dataset (default: ./dataset)')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate existing dataset')
    parser.add_argument('--show_issues', action='store_true',
                       help='Show detailed list of issues found during validation')
    parser.add_argument('--augmentation_config_only', action='store_true',
                       help='Only generate augmentation config file')
    
    args = parser.parse_args()
    
    # Generate augmentation config only
    if args.augmentation_config_only:
        AugmentationConfig.save_augmentation_config()
        return
    
    # Initialize preparer
    preparer = DatasetPreparer(dataset_root=args.dataset_root)
    
    # Validate only mode
    if args.validate_only:
        result = preparer.validate_dataset()
        
        if args.show_issues:
            issues = result['issues']
            if any(issues.values()):
                print("\n📋 DETAILED ISSUES:")
                for issue_type, issue_list in issues.items():
                    if issue_list:
                        print(f"\n{issue_type.replace('_', ' ').title()}:")
                        for issue in issue_list[:10]:  # Show first 10
                            print(f"  - {issue}")
                        if len(issue_list) > 10:
                            print(f"  ... and {len(issue_list) - 10} more")
        return
    
    # Split dataset mode
    if not args.source_images or not args.source_labels:
        parser.error("--source_images and --source_labels are required (unless using --validate_only or --augmentation_config_only)")
    
    print("🚀 Starting dataset preparation...")
    
    # Split dataset
    stats = preparer.split_dataset(
        source_images=args.source_images,
        source_labels=args.source_labels,
        split_ratio=args.split_ratio,
        seed=args.seed
    )
    
    # Validate dataset
    preparer.validate_dataset()
    
    # Generate augmentation config
    AugmentationConfig.save_augmentation_config()
    
    print("\n✅ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Review the dataset at: {args.dataset_root}")
    print(f"2. Update data.yaml if needed")
    print(f"3. Train your model:")
    print(f"   yolo detect train data=data.yaml model=yolov8s.pt epochs=150 imgsz=640")
    print(f"4. Use augmentation_config.yaml for reference")


if __name__ == "__main__":
    main()
