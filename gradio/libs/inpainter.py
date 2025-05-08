from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image
import torch


class SDImpainting:
    def __init__(self, device: str):
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        self.generator = torch.Generator(device=device).manual_seed(0)

        print("Modelo Stable Diffusion XL 1.0 descargado ?")

    def impaint(self, image_path: str, mask_path: str, prompt: str, negative_prompt: str, strength: float, guidance: float):
        # Carga las imágenes originales
        original_image = load_image(image_path)
        original_mask = load_image(mask_path)

        # Guarda las dimensiones originales
        orig_width, orig_height = original_image.size

        # Calcula el factor de escala para ajustar la imagen al canvas de 1024x1024 sin distorsionar la relación de aspecto
        scale = min(1024 / orig_width, 1024 / orig_height)
        new_size = (int(orig_width * scale), int(orig_height * scale))

        # Redimensiona la imagen y la máscara manteniendo la relación de aspecto
        resized_image = original_image.resize(new_size, Image.LANCZOS)
        resized_mask = original_mask.resize(new_size, Image.LANCZOS)

        # Crea un canvas de 1024x1024 para la imagen y la máscara
        canvas_image = Image.new("RGB", (1024, 1024), (255, 255, 255))
        # Ajusta el color de fondo de la máscara según necesites
        canvas_mask = Image.new("RGB", (1024, 1024), (0, 0, 0))

        # Calcula las posiciones para centrar la imagen redimensionada en el canvas
        paste_x = (1024 - new_size[0]) // 2
        paste_y = (1024 - new_size[1]) // 2

        canvas_image.paste(resized_image, (paste_x, paste_y))
        canvas_mask.paste(resized_mask, (paste_x, paste_y))

        # Ejecuta el pipeline de inpainting en el canvas
        result_canvas = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canvas_image,
            mask_image=canvas_mask,
            guidance_scale=guidance,
            strength=strength,
            generator=self.generator,
        ).images[0]

        # Recorta la parte central que contiene la imagen original redimensionada
        result_crop = result_canvas.crop(
            (paste_x, paste_y, paste_x + new_size[0], paste_y + new_size[1]))

        # Reescala la imagen resultante a las dimensiones originales
        final_result = result_crop.resize(
            (orig_width, orig_height), Image.LANCZOS)
        return final_result
