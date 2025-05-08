from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import json


class BLIP:
    def __init__(self, device: str):
        self.device = device
        with open("../secret.json", "r") as f:
            secret = json.load(f)
        
        huggingface_token = secret["hf"]

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", token=huggingface_token)
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", token=huggingface_token).to(self.device)

        print("Modelo BLIP descargado ?")


    def generate_caption(self, image_path: str):
        """Detect what is in the picture"""
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        return f"photo of {self.processor.decode(out[0], skip_special_tokens=True)}"