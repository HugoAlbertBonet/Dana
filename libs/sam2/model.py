import os
import torch
import urllib.request
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import sys
sys.path.append("..")


class SAM2:
    def __init__(self, device: str):
        self.checkpoints_dir = "./checkpoints/"
        model = "sam_vit_h_4b8939"
        self.model_url = f"https://dl.fbaipublicfiles.com/segment_anything/{model}.pth"
        self.model_path = os.path.join(
            self.checkpoints_dir, os.path.basename(self.model_url))

        # Crear el directorio si no existe
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Descargar el archivo si no existe
        if not os.path.exists(self.model_path):
            print(
                f"Descargando el modelo SAM2 {model} desde {self.model_url}...")
            urllib.request.urlretrieve(self.model_url, self.model_path)
            print("Descarga de modelo SAM2 completa.")

        print("Modelo SAM2 descargado ✅")

        sam_checkpoint = f"./checkpoints/{model}.pth"
        model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def get_mask_by_pixel(self, x: float, y: float, image: np.ndarray):
        # Pixel seleccionado
        input_point = np.array([[x, y]])
        # Etiqueta para el punto seleccionado
        input_label = np.array([1])

        # Configurar la imagen en el predictor
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        return masks[0]
    
    def get_mask_by_bounding_box(self, x1: float, y1: float, x2: float, y2: float, image: np.ndarray):
        # Configurar la imagen en el predictor
        input_box = np.array([x1, y1, x2, y2])
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        return masks[0]

    def get_mask_by_bounding_boxes(self, boxes: list, image: np.ndarray):
        self.predictor.set_image(image)
        input_boxes = torch.tensor(boxes, device=self.predictor.device)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
          point_coords=None,
          point_labels=None,
          boxes=transformed_boxes,
          multimask_output=False,
        )

        return masks
