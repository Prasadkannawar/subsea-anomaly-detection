"""
YOLOv8 Training Script for Underwater Anomaly Detection
========================================================

This script trains a YOLOv8 object detection model using transfer learning
from pretrained COCO weights.

Why YOLOv8?
-----------
- Real-time detection: ~30+ FPS on GPU, suitable for live underwater ROV feeds
- Transfer learning: Pretrained on COCO (80 classes), adapts to our 4 underwater classes
- Single-stage detector: Faster than two-stage methods (R-CNN family)
- Anchor-free: Simpler architecture, better generalization
- Built-in augmentations: Robust to underwater lighting/color variations

Transfer Learning Approach:
---------------------------
1. Load yolov8n.pt pretrained on COCO dataset (80 classes)
2. Keep backbone (CSPDarknet) frozen initially - leverages learned features
3. Replace detection head for our 4 classes (corrosion, marine_growth, debris, healthy_surface)
4. Fine-tune entire network on underwater dataset
5. Model learns underwater-specific features while retaining general object detection abilities

Classes:
--------
0: corrosion
1: marine_growth
2: debris
3: healthy_surface

Usage:
------
    python train.py
    
    # Or with custom parameters:
    python train.py --epochs 100 --batch 32 --imgsz 1280

Requirements:
-------------
    pip install ultralytics

Author: NauticAI Team
"""

from ultralytics import YOLO
import argparse
from pathlib import Path


def train_model(
    data_yaml='data.yaml',
    model_name='yolov8n.pt',
    epochs=50,
    imgsz=640,
    batch=16,
    device='0',
    project='runs/train',
    name='underwater_detection'
):
    """
    Train YOLOv8 model for underwater anomaly detection.
    
    Args:
        data_yaml: Path to dataset configuration file
        model_name: Pretrained model to use (yolov8n/s/m/l/x)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size (use -1 for auto)
        device: GPU device (0, 1, etc.) or 'cpu'
        project: Project directory
        name: Experiment name
    """
    
    print("="*60)
    print("🌊 NauticAI - Underwater Anomaly Detection Training")
    print("="*60)
    print(f"Model:        {model_name}")
    print(f"Dataset:      {data_yaml}")
    print(f"Epochs:       {epochs}")
    print(f"Image Size:   {imgsz}")
    print(f"Batch Size:   {batch}")
    print(f"Device:       {device}")
    print("="*60 + "\n")
    
    # Load pretrained YOLOv8 model (transfer learning)
    # This loads weights trained on COCO dataset
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        # Dataset configuration
        data=data_yaml,
        
        # Training parameters
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        
        # Output settings
        project=project,
        name=name,
        exist_ok=False,
        
        # Optimizer (default: SGD with momentum)
        optimizer='auto',  # auto, SGD, Adam, AdamW
        
        # Learning rate
        lr0=0.01,          # Initial learning rate
        lrf=0.01,          # Final learning rate (lr0 * lrf)
        
        # Augmentations (enabled by default in YOLOv8)
        # These simulate underwater conditions and improve generalization
        hsv_h=0.015,       # HSV-Hue augmentation (underwater lighting)
        hsv_s=0.7,         # HSV-Saturation augmentation
        hsv_v=0.4,         # HSV-Value augmentation
        degrees=10.0,      # Rotation augmentation (+/- degrees)
        translate=0.1,     # Translation augmentation
        scale=0.5,         # Scale augmentation
        fliplr=0.5,        # Horizontal flip probability
        mosaic=1.0,        # Mosaic augmentation probability
        mixup=0.1,         # MixUp augmentation probability
        
        # Validation
        val=True,          # Validate during training
        
        # Logging
        plots=True,        # Save training plots
        save=True,         # Save checkpoints
        save_period=-1,    # Save checkpoint every n epochs (-1=disabled)
        
        # Performance
        workers=8,         # Number of data loader workers
        
        # Early stopping
        patience=50,       # Epochs to wait for no improvement
        
        # Model saving
        save_best=True,    # Save best checkpoint
        verbose=True       # Verbose output
    )
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Best model saved to: {project}/{name}/weights/best.pt")
    print(f"Last model saved to: {project}/{name}/weights/last.pt")
    print(f"\nMetrics:")
    print(f"  Best mAP@0.5:      {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Best mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print("="*60 + "\n")
    
    return results


def main():
    """Main training function with CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 for underwater anomaly detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data', type=str, default='data.yaml',
                       help='Path to data.yaml (default: data.yaml)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='Pretrained model (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size, -1 for auto (default: 16)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device: 0, 1, cpu (default: 0)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory (default: runs/train)')
    parser.add_argument('--name', type=str, default='underwater_detection',
                       help='Experiment name (default: underwater_detection)')
    
    args = parser.parse_args()
    
    # Validate data.yaml exists
    if not Path(args.data).exists():
        print(f"❌ Error: {args.data} not found!")
        print("Make sure you have created data.yaml in the project root.")
        return
    
    # Train model
    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )
    
    print("\n🚀 Next Steps:")
    print("1. Evaluate model:")
    print(f"   python evaluate.py --model {args.project}/{args.name}/weights/best.pt")
    print("2. Test inference:")
    print(f"   python predict.py --model {args.project}/{args.name}/weights/best.pt --source path/to/test/image.jpg")
    print("3. Deploy to Streamlit app:")
    print(f"   copy {args.project}\\{args.name}\\weights\\best.pt models\\best.pt")
    print("   streamlit run app.py")


if __name__ == '__main__':
    main()
