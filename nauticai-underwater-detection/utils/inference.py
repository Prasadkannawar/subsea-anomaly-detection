"""
Inference Module

This module handles running YOLOv8 inference on images and videos,
processing detection results, and generating annotated outputs.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple
import tempfile
from datetime import datetime


class InferenceEngine:
    """
    Handles YOLOv8 inference on images and videos.
    """
    
    def __init__(self, model):
        """
        Initialize inference engine with a loaded model.
        
        Args:
            model: Loaded YOLO model object
        """
        self.model = model
        
    def predict_image(self, image_path: Union[str, Path], 
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            dict: Detection results with boxes, classes, scores, and annotated image
        """
        try:
            # Run inference
            results = self.model.predict(
                source=str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                save=False,
                verbose=False
            )
            
            # Extract results
            result = results[0]
            
            # Get annotated image
            annotated_img = result.plot()
            
            # Extract detection details
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'class_id': int(box.cls[0]),
                        'class_name': self.model.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    }
                    detections.append(detection)
            
            return {
                'detections': detections,
                'num_detections': len(detections),
                'annotated_image': annotated_img,
                'original_shape': result.orig_shape,
                'inference_time': result.speed  # preprocessing, inference, postprocessing times
            }
            
        except Exception as e:
            raise Exception(f"Error during image inference: {str(e)}")
    
    def predict_video(self, video_path: Union[str, Path],
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45,
                     output_path: Union[str, Path] = None) -> Dict:
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to input video
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            output_path: Path to save annotated video (optional)
            
        Returns:
            dict: Video detection results and statistics
        """
        try:
            # If no output path provided, create temp file
            if output_path is None:
                output_path = Path('outputs') / f'detected_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            
            # Run inference on video
            results = self.model.predict(
                source=str(video_path),
                conf=conf_threshold,
                iou=iou_threshold,
                save=True,
                project=str(Path(output_path).parent),
                name=Path(output_path).stem,
                verbose=False
            )
            
            # Collect statistics across frames
            total_detections = 0
            unique_classes = set()
            
            for result in results:
                if result.boxes is not None:
                    total_detections += len(result.boxes)
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        unique_classes.add(self.model.names[class_id])
            
            return {
                'output_video_path': str(output_path),
                'total_frames': len(results),
                'total_detections': total_detections,
                'unique_classes': list(unique_classes),
                'avg_detections_per_frame': total_detections / len(results) if results else 0
            }
            
        except Exception as e:
            raise Exception(f"Error during video inference: {str(e)}")
    
    @staticmethod
    def draw_custom_boxes(image: np.ndarray, detections: List[Dict],
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2) -> np.ndarray:
        """
        Draw custom bounding boxes on image.
        
        Args:
            image: Input image (numpy array)
            detections: List of detection dictionaries
            color: Box color in BGR
            thickness: Line thickness
            
        Returns:
            np.ndarray: Image with drawn boxes
        """
        img_copy = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(img_copy, (x1, y1 - label_height - 10),
                         (x1 + label_width, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return img_copy


# Example usage:
if __name__ == "__main__":
    from utils.model_loader import load_default_model
    
    # Load model
    model = load_default_model()
    
    # Create inference engine
    engine = InferenceEngine(model)
    
    # Test image inference
    # results = engine.predict_image('path/to/test/image.jpg')
    # print(f"Found {results['num_detections']} objects")
