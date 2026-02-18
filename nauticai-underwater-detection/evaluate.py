"""
YOLOv8 Model Evaluation Script
===============================

Evaluates trained YOLOv8 model on validation dataset and prints metrics.

Metrics Evaluated:
-----------------
- Precision: What % of detections are correct
- Recall: What % of ground truth objects are detected
- mAP@0.5: Mean Average Precision at IoU threshold 0.5
- mAP@0.5:0.95: mAP averaged across IoU thresholds 0.5-0.95 (COCO metric)

Usage:
------
    python evaluate.py
    
    # Or with custom model:
    python evaluate.py --model runs/train/underwater_detection/weights/best.pt

Author: NauticAI Team
"""

from ultralytics import YOLO
import argparse
from pathlib import Path


def evaluate_model(model_path, data_yaml='data.yaml', imgsz=640, split='val'):
    """
    Evaluate YOLOv8 model on validation dataset.
    
    Args:
        model_path: Path to trained model (.pt file)
        data_yaml: Path to dataset configuration
        imgsz: Image size for validation
        split: Dataset split to evaluate on ('val' or 'test')
    """
    
    print("="*60)
    print("📊 NauticAI - Model Evaluation")
    print("="*60)
    print(f"Model:    {model_path}")
    print(f"Dataset:  {data_yaml}")
    print(f"Split:    {split}")
    print(f"Img Size: {imgsz}")
    print("="*60 + "\n")
    
    # Load trained model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the model file exists and is a valid YOLOv8 model.")
        return None
    
    # Run validation
    print("🔄 Running validation...\n")
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=16,
        conf=0.25,      # Confidence threshold
        iou=0.6,        # NMS IoU threshold
        plots=True,     # Save validation plots
        save_json=False,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("✅ EVALUATION RESULTS")
    print("="*60)
    
    # Overall metrics
    print("\n📈 Overall Performance:")
    print(f"  Precision:     {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall:        {results.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"  mAP@0.5:       {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:0.95:  {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    
    # Per-class metrics
    print("\n📊 Per-Class Metrics:")
    class_names = ['corrosion', 'marine_growth', 'debris', 'healthy_surface']
    
    # Try to extract per-class metrics if available
    if hasattr(results, 'box'):
        if hasattr(results.box, 'ap50'):
            print(f"\n  mAP@0.5 per class:")
            for i, name in enumerate(class_names):
                if i < len(results.box.ap50):
                    print(f"    {name:15}: {results.box.ap50[i]:.4f}")
    
    print("\n" + "="*60)
    print("Validation plots saved to: runs/val/")
    print("="*60 + "\n")
    
    return results


def main():
    """Main evaluation function with CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate YOLOv8 underwater detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, 
                       default='runs/train/underwater_detection/weights/best.pt',
                       help='Path to trained model (default: runs/train/underwater_detection/weights/best.pt)')
    parser.add_argument('--data', type=str, default='data.yaml',
                       help='Path to data.yaml (default: data.yaml)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--split', type=str, default='val',
                       choices=['val', 'test'],
                       help='Dataset split to evaluate (default: val)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        print(f"❌ Error: Model not found at {args.model}")
        print("Please train a model first using: python train.py")
        return
    
    if not Path(args.data).exists():
        print(f"❌ Error: {args.data} not found!")
        return
    
    # Evaluate model
    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        split=args.split
    )


if __name__ == '__main__':
    main()
