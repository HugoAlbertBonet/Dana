import io
import cv2
import os
import gradio as gr
import glob
import torch
import math
import torchvision.transforms as transforms
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from libs.inpainter import SDImpainting
from models import UNet
from libs.face_extractor import FaceExtractor
from utils import (
    generate_binary_mask,
    delete_irrelevant_detected_pixels,
    fill_little_spaces,
    soften_contours,
    blur_mask,
    delete_files
)
import gc

print("All packages imported")
# Cargar variables de entorno
load_dotenv()

# Rutas de archivos generados
RUTA_MASCARA = "processed_mask.png"
RUTA_IMAGEN_FINAL = "final_output.png"
transformations = transforms.Compose([transforms.ToTensor()]) #transforms.Resize((1024, 1024)),

# Configuracion del dispositivo para modelos
DEVICE = "cuda:1"
DEVICE_UNET = "cuda:0"
print(f"DEVICE {DEVICE}")

# Cargar modelos
segmentation_model = UNet(n_class = 1)
impainting_model = SDImpainting(DEVICE)
face_extractor = FaceExtractor()


# Funcion para sacar todos los modelos de segmentacion

def list_models():
    paths = glob.glob("../output/unet*.pth")
    return paths
    
def list_images():
    masks = sorted(glob.glob('../segmentation/masks/*.*'))
    masks = masks[round(0.8*len(masks)):]
    return masks

# Funcion que se ejecuta al cargar una imagen


        
def on_image_load_image(image_path):
      image = Image.open(image_path[:-3].replace("masks", "images")+"jpg").convert('RGB')
      image = transformations(image)
      image = transforms.functional.to_pil_image(image).convert("RGB")
      image.save(f'image.jpg')
      return 'image.jpg'
      
      
def image_resolution(image_path):
      image = Image.open(image_path[:-3].replace("masks", "images")+"jpg").convert('RGB')
      image = transformations(image)
      return str(image.size())
        
def on_image_load_target(image_path):
    real_mask = Image.open(image_path).convert('RGB')
    real_mask = transformations(real_mask)
    real_mask = transforms.functional.to_pil_image(real_mask).convert("RGB")
    real_mask.save(f'target_mask.jpg')
    return 'target_mask.jpg'
        
def on_image_load_pred(image_path, thr):
    image = Image.open(image_path[:-3].replace("masks", "images")+"jpg").convert('RGB')
    image_tensor = transforms.Compose([transforms.Resize((1024, 1024)),transforms.ToTensor()])(image).reshape((1, 3, 1024, 1024)).to(DEVICE_UNET)
    global segmentation_model  
    segmentation_model = segmentation_model.to(DEVICE_UNET)
    mask = segmentation_model(image_tensor)
    mask = torch.sigmoid(mask)
    mask_bin = (mask > thr).float()
    mask_people = face_extractor(image, prompt = "people", return_results = "mask", mask_multiplier = 255)
    mask_people = transforms.Compose([transforms.Resize((1024, 1024)),transforms.ToTensor()])(Image.fromarray(mask_people.astype(np.uint8))).to(DEVICE_UNET)
    mask_bin = mask_bin[0]*(mask_people==0)
    mask = transforms.functional.to_pil_image(mask_bin).convert("RGB")
    mask.save(f'pred_mask.jpg')
    return 'pred_mask.jpg'
    
def on_image_load_pred2(image_path, thr):
    image = Image.open(image_path[:-3].replace("masks", "images")+"jpg").convert('RGB')
    image_tensor = transforms.Compose([transforms.Resize((1024, 1024)),transforms.ToTensor()])(image).reshape((1, 3, 1024, 1024)).to(DEVICE_UNET)
    global segmentation_model  
    segmentation_model = segmentation_model.to(DEVICE_UNET)
    mask = segmentation_model(image_tensor)
    mask = torch.sigmoid(mask)
    mask_bin = (mask > thr).float()
    mask_people = face_extractor(image, prompt = "people", return_results = "mask", mask_multiplier = 255)
    mask_people = transforms.Compose([transforms.Resize((1024, 1024)),transforms.ToTensor()])(Image.fromarray(mask_people.astype(np.uint8))).to(DEVICE_UNET)
    mask_bin = mask_people #mask_bin[0]*(mask_people==0)
    mask = transforms.functional.to_pil_image(mask_bin).convert("RGB")
    mask.save(f'pred_mask.jpg')
    return 'pred_mask.jpg'
    

def extract_false_positives(image_path, target_mask_path, pred_mask_path, thr):
    # Load images
    image = Image.open(image_path)
    image = transformations(image)
    image = transforms.functional.to_pil_image(image).convert("RGB")
    image.save(f'image.jpg')
    image = cv2.imread('image.jpg')
    target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

    # Binarize masks
    _, target_mask_bin = cv2.threshold(pred_mask, int(255*thr), 255, cv2.THRESH_BINARY)
    _, pred_mask_bin = cv2.threshold(pred_mask, int(255*thr), 255, cv2.THRESH_BINARY)
    
    target_mask_bin = target_mask_bin*0

    # Identify false positives
    false_positive_mask = cv2.bitwise_and(pred_mask_bin, cv2.bitwise_not(target_mask_bin))
    false_positive_mask_3ch = cv2.merge([false_positive_mask]*3)

    # Fade the full image
    faded_image = cv2.addWeighted(image, 0.4, np.full_like(image, 255), 0.6, 0)

    # Create a red-tinted overlay for false positives
    red_overlay = image.copy()
    red_overlay[:, :, 0] = 0    # Zero out blue channel
    red_overlay[:, :, 1] = 0    # Zero out green channel
    red_overlay[:, :, 2] = 255  # Max red

    # Blend red overlay with original image in FP regions
    highlighted_fp = cv2.addWeighted(image, 0.3, red_overlay, 0.7, 0)

    # Combine: red-highlighted FP + faded elsewhere
    output_image = np.where(false_positive_mask_3ch == 255, highlighted_fp, faded_image)

    # Save result
    final_path = "false_positives_highlighted_red.png"
    cv2.imwrite(final_path, output_image)

    return final_path

    

def extract_false_negatives(image_path, target_mask_path, pred_mask_path, thr):
    # Load images
    image = Image.open(image_path)
    image = transformations(image)
    image = transforms.functional.to_pil_image(image).convert("RGB")
    image.save(f'image.jpg')
    image = cv2.imread('image.jpg')
    target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

    # Binarize masks
    _, target_mask_bin = cv2.threshold(target_mask, 127, 255, cv2.THRESH_BINARY)
    _, pred_mask_bin = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)

    # Identify false negatives: in target, missing in prediction
    false_negative_mask = cv2.bitwise_and(target_mask_bin, cv2.bitwise_not(pred_mask_bin))
    false_negative_mask_3ch = cv2.merge([false_negative_mask]*3)

    # Fade the entire image
    faded_image = cv2.addWeighted(image, 0.4, np.full_like(image, 255), 0.6, 0)

    # Create blue-tinted overlay
    blue_overlay = image.copy()
    blue_overlay[:, :, 0] = 255  # Max blue
    blue_overlay[:, :, 1] = 0    # Zero green
    blue_overlay[:, :, 2] = 0    # Zero red

    # Blend overlay with image in FN areas
    highlighted_fn = cv2.addWeighted(image, 0.3, blue_overlay, 0.7, 0)

    # Combine: blue-highlighted FN + faded elsewhere
    output_image = np.where(false_negative_mask_3ch == 255, highlighted_fn, faded_image)

    # Save result
    final_path = "false_negatives_highlighted_blue.png"
    cv2.imwrite(final_path, output_image)

    return final_path



# Funcion para aplicar fill_little_spaces al reenviar de la 3era a la 4ta imagen


def apply_fill_little_spaces(image_path, mask_type):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("No se pudo cargar la imagen.")
        processed = delete_irrelevant_detected_pixels(image)
        processed = fill_little_spaces(processed)
        processed = soften_contours(processed)
        output_path = f"softened_contours_{mask_type}.png"
        cv2.imwrite(output_path, processed)
        return output_path
    except Exception as e:
        print(f"Error en apply_fill_little_spaces: {e}")
        return None

# Procesar la imagen final usando la imagen original y la quinta imagen
def process_final_image_patches(original_image_path, target_image_path, text, strength, guidance, negative_prompt):
    try:
        target_image_path = apply_fill_little_spaces(target_image_path, "target")
        image = Image.open(original_image_path)
        image = transforms.ToTensor()(image)
        mask = Image.open(target_image_path)
        mask = transforms.Compose([transforms.Resize((image.shape[1], image.shape[2])),transforms.ToTensor()])(mask)
        counts = torch.zeros(3, image.shape[1], image.shape[2])
        final_image = torch.zeros(3, image.shape[1], image.shape[2])
        nw = math.ceil(image.shape[1]/1024)
        nh = math.ceil(image.shape[2]/1024)
        stridew = (image.shape[1] - 1024) // (nw-1)
        strideh = (image.shape[2] - 1024) // (nh-1)
        for i in range(nw):
          for j in range(nh):
            centerw = 511 + stridew*i
            centerh = 511 + strideh*j
            image_patch = image[:, centerw-511:centerw+513, centerh-511:centerh+513]
            mask_patch = mask[:, centerw-511:centerw+513, centerh-511:centerh+513]
            print(torch.sum(mask_patch))
            #print(mask_patch)
            if torch.sum(mask_patch).item() > 0:
              image_pil = transforms.functional.to_pil_image(image_patch).convert("RGB")
              image_pil.save(f'{original_image_path}_patch{i},{j}.jpg')
              mask_pil = transforms.functional.to_pil_image(mask_patch).convert("RGB")
              mask_pil.save(f'{target_image_path}_patch{i},{j}_mask.jpg')
              torch.cuda.empty_cache()
              gc.collect()
              new_image_patch = impainting_model.impaint(
                  image_path=f'{original_image_path}_patch{i},{j}.jpg',
                  mask_path=f'{target_image_path}_patch{i},{j}_mask.jpg',
                  prompt="",
                  strength=strength,
                  guidance=guidance,
                  negative_prompt=negative_prompt
              )
              new_image_patch = transforms.ToTensor()(new_image_patch)
              final_image[:, centerw-511:centerw+513, centerh-511:centerh+513] += new_image_patch
              image[:, centerw-511:centerw+513, centerh-511:centerh+513] = image[:, centerw-511:centerw+513, centerh-511:centerh+513]*(mask_patch==0) + new_image_patch*(mask_patch>0)
              counts[:, centerw-511:centerw+513, centerh-511:centerh+513] += torch.ones(3,1024,1024)*(mask_patch>0)
            else: 
              pass
        new_image = image #final_image/counts
        new_image = transforms.functional.to_pil_image(new_image).convert("RGB")
        new_image.save("final_output_target.png")
        return "final_output_target.png"
    except Exception as e:
        print(f"Error: {e}")
        return None
        
        
        
def process_final_image_pred(original_image_path, pred_image_path, text, strength, guidance, negative_prompt):
    try:
        pred_image_path = apply_fill_little_spaces(pred_image_path, "pred")
        torch.cuda.empty_cache()
        gc.collect()
        new_image = impainting_model.impaint(
            image_path=original_image_path,
            mask_path=pred_image_path,
            prompt="",
            strength=strength,
            guidance=guidance,
            negative_prompt=negative_prompt
        )
        new_image.save("final_output_pred.png")
        return "final_output_pred.png"
    except Exception as e:
        print(f"Error: {e}")
        return None

def upload_model(path):
    print(f"Se cambia a modelo {path}")  
    
    torch.cuda.empty_cache()
    gc.collect()
    global segmentation_model  
    segmentation_model.load_state_dict(torch.load(f"{path}", weights_only=True))
    segmentation_model= segmentation_model.to(DEVICE_UNET)

# Construccion de la interfaz en Gradio
with gr.Blocks() as demo:

    # Modelo para segmentar
    with gr.Row():
        text_input = gr.Textbox(label="Enter prompt",
                                placeholder="Write prompt for impainting...")
        mask_button = gr.Button("Extract Mask")
        threshold = gr.Slider(minimum=0.0, maximum=1.0,
                             value=0.5, label="Threshold", interactive=True)
    with gr.Row():
        model_path = gr.Dropdown(choices=list_models(), label="Modelos disponibles", scale=4)
        image_path = gr.Dropdown(choices=list_images(), label="Imagenes disponibles", scale=4)
        
        
    # Fila 1: Imagen de entrada, mascara real y mascara predicha
    with gr.Row():
        img = gr.Image(label="Input Image", type="filepath", interactive = True)
        target = gr.Image(label="Target Mask", type="filepath")
        pred = gr.Image(label="Predicted Mask", type="filepath")
    

    # Boton para generar imagenes con ambas mascaras
    with gr.Row():
        send_button_patches = gr.Button("Generate image from patches (slow)")
        send_button_pred = gr.Button("Generate image with pred mask (fast)")
        
    with gr.Row():
        strength = gr.Slider(minimum=0.0, maximum=1.0,
                             value=0.99, label="Strength", interactive=True)
        guidance = gr.Slider(minimum=0.0, maximum=50.0,
                             value=7.0, label="Guidance Scale", interactive=True)
                             
    with gr.Row():
        negative_prompt = gr.Textbox(
            label="Negative prompt", placeholder="Write negative prompt...")

    # Fila 3: Inpainting con ambas mascaras (con y sin dilatado?)
    with gr.Row():
        img_target = gr.Image(label="With target", type="filepath")
    with gr.Row():
        img_pred = gr.Image(label="With predicted", type="filepath")


    # Al cargar la imagen se genera el caption en el textbox
    image_path.change(on_image_load_image, inputs=[image_path], outputs=img)
    image_path.change(on_image_load_target, inputs=[image_path], outputs=target)
    image_path.change(on_image_load_pred, inputs=[image_path, threshold], outputs=pred)
    image_path.change(image_resolution, inputs=[image_path], outputs=text_input)
    mask_button.click(on_image_load_pred2, inputs=[img, threshold], outputs=pred)
    model_path.change(fn=upload_model, inputs=model_path, outputs=None)


    # Reiniciar la mascara al cambiar de imagen
    def reset_mask(image_path):
        delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])
        return None

    # Asignar eventos a la interfaz
    #img.change(reset_mask, inputs=[img], outputs=None)

    # Boton para generar la imagen final usando la imagen original y las mascaras
    send_button_patches.click(process_final_image_patches, inputs=[
                      img, pred, text_input, strength, guidance, negative_prompt], outputs=img_target)
    
    send_button_pred.click(process_final_image_pred, inputs=[
                      img, pred, text_input, strength, guidance, negative_prompt], outputs=img_pred)

# Limpiar archivos previos antes de lanzar la aplicacion
delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])

# Lanzar la interfaz
demo.launch(debug=True)
