"""
SUIM Segmentation to YOLO Bounding Box Converter
=================================================

Converts SUIM underwater segmentation masks into YOLOv8 bounding box format.

SUIM Classes (8 total):
    0 - BW: Background/waterbody (skip)
    1 - HD: Human divers (skip)
    2 - PF: Aquatic plants/sea-grass
    3 - WR: Wrecks/ruins
    4 - RO: Robots/instruments (skip)
    5 - RI: Reefs/invertebrates
    6 - FV: Fish/vertebrates (skip)
    7 - SR: Sea-floor/rocks

Our Class Mapping (4 classes):
    0: corrosion (from WR - wrecks/ruins)
    1: marine_growth (from PF, RI - plants, reefs)
    2: debris (from WR - some ruins can be debris)
    3: healthy_surface (from SR - sea-floor/rocks)

Usage:
    python convert_suim_to_yolo.py --images_dir ./SUIM_FULL/train_val/images --masks_dir ./SUIM_FULL/train_val/masks --output_dir ./raw_data

Author: NauticAI Team
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


class SUIMConverter:
    """Converts SUIM segmentation masks to YOLO bounding boxes."""
    
    # SUIM class ID -> Our class (ID, name)
    CLASS_MAP = {
        2: (1, 'marine_growth'),  # PF: Plants
        3: (0, 'corrosion'),      # WR: Wrecks/ruins
        5: (1, 'marine_growth'),  # RI: Reefs
        7: (3, 'healthy_surface'), # SR: Sea-floor
    }
    
    def __init__(self, min_area=100, min_width=10, min_height=10):
        """
        Initialize converter.
        
        Args:
            min_area: Minimum bounding box area (pixels)
            min_width: Minimum bbox width (pixels)
            min_height: Minimum bbox height (pixels)
        """
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.stats = {
            'images_processed': 0,
            'images_with_objects': 0,
            'total_boxes': 0,
            'boxes_per_class': {0: 0, 1: 0, 2: 0, 3: 0},
            'skipped_small': 0
        }
    
    def mask_to_bboxes(self, mask_path, img_width, img_height):
        """
        Convert segmentation mask to YOLO bounding boxes.
        
        Args:
            mask_path: Path to mask image
            img_width: Original image width
            img_height: Original image height
            
        Returns:
            list: YOLO format boxes [(class_id, x_center, y_center, w, h), ...]
        """
        # Read mask (grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"⚠ Warning: Could not read {mask_path}")
            return []
        
        boxes = []
        
        # Process each SUIM class we care about
        for suim_class, (our_class, class_name) in self.CLASS_MAP.items():
            # Create binary mask for this class
            binary_mask = (mask == suim_class).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Convert each contour to bbox
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter small boxes
                if area < self.min_area or w < self.min_width or h < self.min_height:
                    self.stats['skipped_small'] += 1
                    continue
                
                # Convert to YOLO format (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                
                # Clamp to [0, 1]
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                norm_w = np.clip(norm_w, 0, 1)
                norm_h = np.clip(norm_h, 0, 1)
                
                boxes.append((our_class, x_center, y_center, norm_w, norm_h))
                
                # Update stats
                self.stats['total_boxes'] += 1
                self.stats['boxes_per_class'][our_class] += 1
        
        return boxes
    
    def convert_dataset(self, images_dir, masks_dir, output_dir):
        """
        Convert entire SUIM dataset.
        
        Args:
            images_dir: Directory with images
            masks_dir: Directory with masks
            output_dir: Output directory (will create images/ and labels/)
        """
        images_path = Path(images_dir)
        masks_path = Path(masks_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        output_images = output_path / 'images'
        output_labels = output_path / 'labels'
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        # Get image files
        image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_exts:
            image_files.extend(images_path.glob(ext))
        
        print(f"\n🚀 Converting {len(image_files)} images...")
        print(f"   Source images: {images_dir}")
        print(f"   Source masks:  {masks_dir}")
        print(f"   Output:        {output_dir}\n")
        
        # Process each image
        for idx, img_file in enumerate(image_files, 1):
            self.stats['images_processed'] += 1
            
            # Find corresponding mask
            mask_file = masks_path / f"{img_file.stem}.bmp"
            if not mask_file.exists():
                mask_file = masks_path / f"{img_file.stem}.png"
            
            if not mask_file.exists():
                print(f"⚠ No mask for {img_file.name}")
                continue
            
            # Get image dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"⚠ Could not read {img_file.name}")
                continue
            
            img_h, img_w = img.shape[:2]
            
            # Convert mask to bboxes
            boxes = self.mask_to_bboxes(mask_file, img_w, img_h)
            
            if not boxes:
                continue
            
            self.stats['images_with_objects'] += 1
            
            # Copy image to output
            output_img = output_images / img_file.name
            cv2.imwrite(str(output_img), img)
            
            # Save YOLO labels
            label_file = output_labels / f"{img_file.stem}.txt"
            with open(label_file, 'w') as f:
                for class_id, x_c, y_c, w, h in boxes:
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            # Progress
            if idx % 50 == 0 or idx == len(image_files):
                print(f"   Processed: {idx}/{len(image_files)} images")
        
        self.print_summary()
    
    def print_summary(self):
        """Print conversion summary."""
        print("\n" + "="*60)
        print("📊 CONVERSION SUMMARY")
        print("="*60)
        print(f"Images Processed:        {self.stats['images_processed']}")
        print(f"Images with Objects:     {self.stats['images_with_objects']}")
        print(f"Total Bounding Boxes:    {self.stats['total_boxes']}")
        print(f"Skipped (too small):     {self.stats['skipped_small']}")
        print(f"\nClass Distribution:")
        class_names = {
            0: 'corrosion',
            1: 'marine_growth', 
            2: 'debris',
            3: 'healthy_surface'
        }
        for class_id in sorted(self.stats['boxes_per_class'].keys()):
            count = self.stats['boxes_per_class'][class_id]
            name = class_names[class_id]
            print(f"  Class {class_id} ({name:15}): {count:5} boxes")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert SUIM masks to YOLO bounding boxes',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing SUIM images')
    parser.add_argument('--masks_dir', type=str, required=True,
                       help='Directory containing SUIM masks')
    parser.add_argument('--output_dir', type=str, default='./raw_data',
                       help='Output directory (default: ./raw_data)')
    parser.add_argument('--min_area', type=int, default=100,
                       help='Minimum bbox area in pixels (default: 100)')
    parser.add_argument('--min_width', type=int, default=10,
                       help='Minimum bbox width in pixels (default: 10)')
    parser.add_argument('--min_height', type=int, default=10,
                       help='Minimum bbox height in pixels (default: 10)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.images_dir).exists():
        print(f"❌ Error: Images directory not found: {args.images_dir}")
        return
    
    if not Path(args.masks_dir).exists():
        print(f"❌ Error: Masks directory not found: {args.masks_dir}")
        return
    
    # Convert
    converter = SUIMConverter(
        min_area=args.min_area,
        min_width=args.min_width,
        min_height=args.min_height
    )
    
    converter.convert_dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir
    )
    
    print("✅ Conversion complete!\n")
    print("Next steps:")
    print(f"1. Prepare dataset:")
    print(f"   cd nauticai-underwater-detection")
    print(f"   python dataset_prepare.py --source_images {args.output_dir}/images --source_labels {args.output_dir}/labels")
    print(f"2. Train model:")
    print(f"   yolo detect train data=data.yaml model=yolov8s.pt epochs=150 imgsz=640")


if __name__ == '__main__':
    main()
