"""
SUIM Mask to YOLO Bounding Box Converter
==========================================

This script converts SUIM segmentation masks to YOLO format bounding boxes.

SUIM Dataset uses semantic segmentation masks with these classes:
- BW (0): Background/waterbody (skip this)
- HD (1): Human divers
- PF (2): Aquatic plants and sea-grass
- WR (3): Wrecks/ruins
- RO (4): Robots/instruments
- RI (5): Reefs/invertebrates
- FV (6): Fish and vertebrates
- SR (7): Sea-floor/rocks

Output Classes for Object Detection (mapping):
- 0: corrosion (mapped from WR - wrecks/ruins showing degradation)
- 1: marine_growth (mapped from PF, RI - plants and reefs)
- 2: debris (mapped from RO, WR - robots and ruins)
- 3: healthy_surface (mapped from SR - sea-floor/rocks)

Usage:
------
    python suim_to_yolo.py --suim_root ./SUIM/data/test --output_dir ./raw_data

Requirements:
-------------
    pip install opencv-python numpy pillow

Author: NauticAI Team
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


class SUIMtoYOLOConverter:
    """
    Converts SUIM segmentation masks to YOLO bounding boxes.
    """
    
    # SUIM class IDs to our detection classes mapping
    CLASS_MAPPING = {
        # SUIM class: (our_class_id, class_name)
        3: (0, 'corrosion'),      # WR (wrecks/ruins) -> corrosion
        2: (1, 'marine_growth'),  # PF (plants) -> marine_growth
        5: (1, 'marine_growth'),  # RI (reefs) -> marine_growth
        4: (2, 'debris'),         # RO (robots) -> debris
        7: (3, 'healthy_surface'), # SR (sea-floor) -> healthy_surface
        # Skip: 0 (background), 1 (human divers), 6 (fish)
    }
    
    def __init__(self, min_area: int = 100, min_dim: int = 10):
        """
        Initialize converter.
        
        Args:
            min_area: Minimum bounding box area (pixels)
            min_dim: Minimum bounding box width/height (pixels)
        """
        self.min_area = min_area
        self.min_dim = min_dim
        self.stats = {
            'total_images': 0,
            'total_boxes': 0,
            'skipped_small': 0,
            'class_counts': {}
        }
    
    def mask_to_bboxes(self, mask_path: Path) -> list:
        """
        Extract bounding boxes from segmentation mask.
        
        Args:
            mask_path: Path to mask image
            
        Returns:
            list: List of (class_id, x_center, y_center, width, height) tuples
        """
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠ Warning: Could not read mask {mask_path}")
            return []
        
        h, w = mask.shape
        bboxes = []
        
        # Process each SUIM class
        for suim_class_id, (our_class_id, class_name) in self.CLASS_MAPPING.items():
            # Create binary mask for this class
            class_mask = (mask == suim_class_id).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(
                class_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Convert each contour to bounding box
            for contour in contours:
                x, y, box_w, box_h = cv2.boundingRect(contour)
                area = box_w * box_h
                
                # Filter small boxes
                if area < self.min_area or box_w < self.min_dim or box_h < self.min_dim:
                    self.stats['skipped_small'] += 1
                    continue
                
                # Convert to YOLO format (normalized)
                x_center = (x + box_w / 2) / w
                y_center = (y + box_h / 2) / h
                norm_width = box_w / w
                norm_height = box_h / h
                
                # Ensure values are in [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                bboxes.append((our_class_id, x_center, y_center, norm_width, norm_height))
                
                # Update stats
                self.stats['total_boxes'] += 1
                self.stats['class_counts'][our_class_id] = \
                    self.stats['class_counts'].get(our_class_id, 0) + 1
        
        return bboxes
    
    def convert_dataset(self, 
                       suim_images_dir: Path,
                       suim_masks_dir: Path,
                       output_images_dir: Path,
                       output_labels_dir: Path):
        """
        Convert entire SUIM dataset to YOLO format.
        
        Args:
            suim_images_dir: SUIM images directory
            suim_masks_dir: SUIM masks directory
            output_images_dir: Output images directory
            output_labels_dir: Output labels directory
        """
        # Create output directories
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in suim_images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"\n🔄 Converting {len(image_files)} images from SUIM to YOLO format...")
        
        images_with_boxes = 0
        
        # Process each image
        for img_file in tqdm(image_files, desc="Processing"):
            self.stats['total_images'] += 1
            
            # Find corresponding mask
            mask_file = suim_masks_dir / f"{img_file.stem}.bmp"
            if not mask_file.exists():
                # Try .png extension
                mask_file = suim_masks_dir / f"{img_file.stem}.png"
            
            if not mask_file.exists():
                print(f"⚠ Warning: No mask found for {img_file.name}")
                continue
            
            # Extract bounding boxes
            bboxes = self.mask_to_bboxes(mask_file)
            
            if not bboxes:
                # Skip images with no valid boxes
                continue
            
            images_with_boxes += 1
            
            # Copy image
            shutil.copy2(img_file, output_images_dir / img_file.name)
            
            # Write YOLO label file
            label_file = output_labels_dir / f"{img_file.stem}.txt"
            with open(label_file, 'w') as f:
                for class_id, x_center, y_center, width, height in bboxes:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        self._print_summary(images_with_boxes)
    
    def _print_summary(self, images_with_boxes: int):
        """Print conversion summary."""
        print("\n" + "="*60)
        print("📊 CONVERSION SUMMARY")
        print("="*60)
        print(f"Total Images Processed:  {self.stats['total_images']}")
        print(f"Images with Boxes:       {images_with_boxes}")
        print(f"Total Bounding Boxes:    {self.stats['total_boxes']}")
        print(f"Boxes Skipped (small):   {self.stats['skipped_small']}")
        print(f"\nClass Distribution:")
        
        class_names = {0: 'corrosion', 1: 'marine_growth', 2: 'debris', 3: 'healthy_surface'}
        for class_id in sorted(self.stats['class_counts'].keys()):
            count = self.stats['class_counts'][class_id]
            name = class_names.get(class_id, 'unknown')
            print(f"  Class {class_id} ({name}): {count} boxes")
        
        print("="*60 + "\n")


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description="Convert SUIM segmentation masks to YOLO bounding boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert SUIM test set
  python suim_to_yolo.py --suim_root ./SUIM/data/test --output_dir ./raw_data
  
  # With custom parameters
  python suim_to_yolo.py --suim_root ./SUIM/data/test --output_dir ./raw_data --min_area 200 --min_dim 15
        """
    )
    
    parser.add_argument('--suim_root', type=str, required=True,
                       help='Path to SUIM dataset root (containing images/ and masks/)')
    parser.add_argument('--output_dir', type=str, default='./raw_data',
                       help='Output directory for YOLO format data (default: ./raw_data)')
    parser.add_argument('--min_area', type=int, default=100,
                       help='Minimum bounding box area in pixels (default: 100)')
    parser.add_argument('--min_dim', type=int, default=10,
                       help='Minimum bounding box width/height in pixels (default: 10)')
    
    args = parser.parse_args()
    
    # Setup paths
    suim_root = Path(args.suim_root)
    suim_images_dir = suim_root / "images"
    suim_masks_dir = suim_root / "masks"
    
    output_dir = Path(args.output_dir)
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    
    # Validate input
    if not suim_images_dir.exists():
        print(f"❌ Error: SUIM images directory not found: {suim_images_dir}")
        return
    
    if not suim_masks_dir.exists():
        print(f"❌ Error: SUIM masks directory not found: {suim_masks_dir}")
        return
    
    print("🚀 Starting SUIM to YOLO conversion...")
    print(f"   Source: {suim_root}")
    print(f"   Output: {output_dir}")
    
    # Convert
    converter = SUIMtoYOLOConverter(
        min_area=args.min_area,
        min_dim=args.min_dim
    )
    
    converter.convert_dataset(
        suim_images_dir=suim_images_dir,
        suim_masks_dir=suim_masks_dir,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir
    )
    
    print("✅ Conversion complete!")
    print(f"\nNext steps:")
    print(f"1. Prepare dataset for training:")
    print(f"   python dataset_prepare.py --source_images {output_images_dir} --source_labels {output_labels_dir}")
    print(f"2. Validate the dataset:")
    print(f"   python dataset_prepare.py --validate_only")
    print(f"3. Train your model:")
    print(f"   yolo detect train data=data.yaml model=yolov8s.pt epochs=150 imgsz=640")


if __name__ == "__main__":
    main()
