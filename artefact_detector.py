"""
artefact_detector.py - AI object detector (AI only, no traditional fallback)
"""

import numpy as np
import cv2
from ultralytics import YOLO

class ArtefactDetector:
    def __init__(self, model_path="artefact_detector/exp1/weights/best.pt", conf_threshold=0.3):
        """
        Initialize detector
        conf_threshold: confidence threshold for AI detection (default 0.3, can be lowered to increase detection rate)
        """
        self.conf_threshold = conf_threshold
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
            print(f"AI model loaded: {model_path}")
            print(f"Recognizable classes: {self.model.names}")
            print(f"Detection confidence threshold: {self.conf_threshold}")
        except Exception as e:
            print(f"Failed to load AI model: {e}")
            self.model_loaded = False
        
        self.class_names = {0: 'cross', 1: 'fiducial', 2: 'circle', 3: 'square'}
    
    def detect(self, image, conf_threshold=None):
        """AI only detection, no traditional method"""
        if not self.model_loaded:
            return []
        
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        results = self.model(image, verbose=False, conf=conf_threshold)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    
                    detections.append({
                        'method': 'ai',
                        'class_id': cls,
                        'class_name': self.class_names.get(cls, 'unknown'),
                        'center': (center_x, center_y),
                        'bbox': xyxy,
                        'confidence': conf
                    })
        return detections
    
    def detect_in_fov(self, fov_image, fov_x, fov_y, conf_threshold=None):
        detections = self.detect(fov_image, conf_threshold=conf_threshold)
        results = []
        for det in detections:
            cx, cy = det['center']
            results.append({
                **det,
                'abs_coord': (fov_x + cx, fov_y + cy)
            })
        return results
    
    def detect_in_scan(self, state, artifact_layer, show_artifact, 
                       half_range=500, step=100, conf_threshold=None):
        from afm_utils import create_fov_image
        
        start_x = max(0, state.x - half_range)
        end_x = min(state.width_um - state.fov_width, state.x + half_range)
        start_y = max(0, state.y - half_range)
        end_y = min(state.height_um - state.fov_height, state.y + half_range)
        
        detected = {}
        
        print(f"Scan range: X=[{start_x:.0f}, {end_x:.0f}], Y=[{start_y:.0f}, {end_y:.0f}]")
        
        for cx in np.arange(start_x, end_x, step):
            for cy in np.arange(start_y, end_y, step):
                fov, _, _ = create_fov_image(
                    state.sample, artifact_layer, show_artifact,
                    cx, cy, state.fov_width, state.fov_height
                )
                detections = self.detect_in_fov(fov, cx, cy, conf_threshold=conf_threshold)
                
                for det in detections:
                    # Use class name + center coordinates as a rough unique ID (a more stable ID may be needed in practice)
                    artefact_id = f"{det['class_name']}_{det['center'][0]:.0f}_{det['center'][1]:.0f}"
                    detected[artefact_id] = det['abs_coord']
                    print(f"  Detected: {det['class_name']} (confidence: {det['confidence']:.2f}) at ({det['abs_coord'][0]:.1f}, {det['abs_coord'][1]:.1f})")
        
        return detected