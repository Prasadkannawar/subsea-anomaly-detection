"""
YOLOv8 Inference Test Script
=============================

Runs inference on test images/videos using trained model.
Saves annotated results for visual inspection.

Usage:
------
    # Single image
    python predict.py --source path/to/image.jpg
    
    # Multiple images
    python predict.py --source path/to/images/
    
    # Video
    python predict.py --source path/to/video.mp4
    
    # Webcam
    python predict.py --source 0
    
    # Custom model
    python predict.py --model runs/train/exp/weights/best.pt --source test.jpg

Author: NauticAI Team
"""

from ultralytics import YOLO
import argparse
from pathlib import Path
import cv2


def predict(
    model_path,
    source,
    conf_threshold=0.25,
    iou_threshold=0.45,
    imgsz=640,
    save=True,
    show=False
):
    """
    Run YOLOv8 inference on images or videos.
    
    Args:
        model_path: Path to trained model (.pt file)
        source: Path to image/video/directory or webcam (0)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        imgsz: Image size for inference
        save: Save annotated results
        show: Display results in window
    """
    
    print("="*60)
    print("🔍 NauticAI - Inference Test")
    print("="*60)
    print(f"Model:      {model_path}")
    print(f"Source:     {source}")
    print(f"Confidence: {conf_threshold}")
    print(f"IoU:        {iou_threshold}")
    print(f"Image Size: {imgsz}")
    print("="*60 + "\n")
    
    # Load trained model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Run inference
    print("🔄 Running inference...\n")
    
    results = model.predict(
        source=source,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        save=save,
        save_txt=True,     # Save labels
        save_conf=True,    # Save confidence in labels
        show=show,
        project='runs/predict',
        name='underwater_test',
        exist_ok=True,
        verbose=True
    )
    
    # Print detection summary
    print("\n" + "="*60)
    print("✅ INFERENCE COMPLETE")
    print("="*60)
    
    total_detections = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    class_names = {
        0: 'corrosion',
        1: 'marine_growth',
        2: 'debris',
        3: 'healthy_surface'
    }
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_counts[class_id] += 1
                total_detections += 1
    
    print(f"\nTotal Detections: {total_detections}")
    print(f"\nDetections by Class:")
    for class_id, count in class_counts.items():
        if count > 0:
            print(f"  {class_names[class_id]:15}: {count}")
    
    if save:
        print(f"\n📁 Results saved to: runs/predict/underwater_test/")
        print("   - Annotated images")
        print("   - Label files (.txt)")
    
    print("="*60 + "\n")
    
    return results


def main():
    """Main prediction function with CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Run YOLOv8 inference on underwater images/videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on single image
  python predict.py --source test_images/underwater001.jpg
  
  # Predict on directory of images
  python predict.py --source test_images/
  
  # Predict on video
  python predict.py --source underwater_video.mp4
  
  # Use custom model with lower confidence
  python predict.py --model models/best.pt --source test.jpg --conf 0.15
        """
    )
    
    parser.add_argument('--model', type=str,
                       default='runs/train/underwater_detection/weights/best.pt',
                       help='Path to trained model (default: runs/train/underwater_detection/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True,
                       help='Source: image file, video file, directory, or webcam (0)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU threshold (default: 0.45)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results (default: True)')
    parser.add_argument('--show', action='store_true', default=False,
                       help='Display results in window')
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        print(f"❌ Error: Model not found at {args.model}")
        print("Please train a model first using: python train.py")
        print("Or specify correct model path with --model")
        return
    
    # Run prediction
    predict(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show
    )
    
    print("🚀 Next Steps:")
    print("1. Review annotated images in runs/predict/underwater_test/")
    print("2. Adjust --conf threshold if needed (lower=more detections, higher=more precise)")
    print("3. Deploy to production:")
    print("   copy runs\\train\\underwater_detection\\weights\\best.pt models\\best.pt")
    print("   streamlit run app.py")


if __name__ == '__main__':
    main()
