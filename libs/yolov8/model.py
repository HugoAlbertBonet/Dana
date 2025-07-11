import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

#device = torch.device('cuda:0')

class YOLOV8:
    def __init__(self, model_path: str=None, device: str="cuda:0"):
      torch.cuda.empty_cache()
      torch.device(device)
      self.device = device
      self.model = None
      if model_path is not None:
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def set_model(self, model_path):
      torch.cuda.empty_cache()  # Limpia memoria antes de cargar nuevo modelo
      self.model = YOLO(model_path)
      self.model.to(self.device)
    
    def get_bounding_box(self, confidence: float, image_path: str):
      torch.cuda.empty_cache()
      results = self.model.predict(source=image_path, save=False, conf=confidence, verbose=False)
    
      # Imagen con bounding boxes dibujadas
      annotated_frame = results[0].plot()
      annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
      image_with_boxes = Image.fromarray(annotated_frame_rgb)

      # Obtener bounding boxes
      boxes = results[0].boxes
      coordinates = boxes.xyxy.cpu().numpy()     # (x1, y1, x2, y2)
      bbox_coords = [list(map(float, coord)) for coord in coordinates]

      return image_with_boxes, bbox_coords
