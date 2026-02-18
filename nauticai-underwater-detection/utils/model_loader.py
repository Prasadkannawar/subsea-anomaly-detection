"""
Model Loader Module

This module handles loading pre-trained YOLOv8 models for underwater anomaly detection.
It supports loading models from local files or Ultralytics hub.
"""

from ultralytics import YOLO
import os
from pathlib import Path


class ModelLoader:
    """
    Handles loading and caching YOLOv8 models.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the model loader.
        
        Args:
            model_path (str): Path to the trained model file (.pt)
                            If None, will use default pre-trained model
        """
        self.model_path = model_path
        self.model = None
        
    def load_model(self, model_path=None):
        """
        Load a YOLOv8 model from disk or download pre-trained model.
        
        Args:
            model_path (str, optional): Path to model file. 
                                       If None, uses self.model_path or default.
        
        Returns:
            YOLO: Loaded YOLO model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        try:
            # Use provided path, instance path, or default
            path = model_path or self.model_path
            
            if path and Path(path).exists():
                print(f"Loading custom model from: {path}")
                self.model = YOLO(path)
            else:
                # For demo/testing: Load pre-trained YOLOv8 model
                # Replace 'yolov8n.pt' with your trained model in production
                print("Loading default YOLOv8n model (for testing only)")
                print("WARNING: Train a custom model for underwater detection!")
                self.model = YOLO('yolov8n.pt')
                
            return self.model
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including class names and count
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
            
        return {
            'names': self.model.names,
            'num_classes': len(self.model.names),
            'task': 'detect'
        }


def load_default_model(model_dir='models'):
    """
    Convenience function to load the default trained model.
    
    Args:
        model_dir (str): Directory containing trained models
        
    Returns:
        YOLO: Loaded model
    """
    # Look for trained model in models directory
    model_path = Path(model_dir) / 'best.pt'
    
    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        print("Using default YOLOv8n model. Train a custom model first!")
        model_path = None
    
    loader = ModelLoader(str(model_path) if model_path else None)
    return loader.load_model()


# Example usage:
if __name__ == "__main__":
    # Test model loading
    loader = ModelLoader()
    model = loader.load_model()
    info = loader.get_model_info()
    print(f"Model loaded successfully!")
    print(f"Classes: {info['names']}")
    print(f"Number of classes: {info['num_classes']}")
