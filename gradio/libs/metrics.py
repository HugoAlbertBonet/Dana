import torch
from PIL import Image
import requests
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import torchvision.transforms as T
import gc
from sklearn.metrics.pairwise import cosine_similarity


class reMOVE():
  def __init__(self, device = "cpu"):
    self.model = sam_model_registry["vit_b"](checkpoint="./libs/sam_vit_b_01ec64.pth").to(device)
    self.model.eval()
    self.device = device
    
  def __call__(self, mask_path, image_dirt_path, image_clean_path):
    gc.collect()
    mask = T.Resize((1024, 1024))(T.ToTensor()(Image.open(mask_path).convert('L')))
    image_dirt = Image.open(image_dirt_path).resize((1024, 1024)).convert('RGB')
    image_clean = Image.open(image_clean_path).resize((1024, 1024)).convert('RGB')
  
    # Load processor and model
    transform = T.Resize((1024, 1024))
    image_clean = transform(image_clean)
    image_dirt = transform(image_dirt)
  
    # Convert to tensor
    to_tensor = T.ToTensor()
    inputs_clean = to_tensor(image_clean).unsqueeze(0).to(self.device)  # (1, 3, 1024, 1024)
    inputs_dirt = to_tensor(image_dirt).unsqueeze(0).to(self.device)  # (1, 3, 1024, 1024)
  
    # Get per-pixel embedding from image encoder
    with torch.no_grad():
        outputs_clean = self.model.image_encoder(inputs_clean)
        outputs_dirt = self.model.image_encoder(inputs_dirt)
  
    patch_embeddings_clean = outputs_clean
    patch_embeddings_dirt = outputs_dirt
    
    # Optionally upsample to image size to approximate per-pixel embeddings
    upsampled_clean = torch.nn.functional.interpolate(patch_embeddings_clean, size=(1024, 1024), mode='bilinear').to(self.device)
    upsampled_dirt = torch.nn.functional.interpolate(patch_embeddings_dirt, size=(1024, 1024), mode='bilinear').to(self.device)
    upsampled_clean.shape, upsampled_dirt.shape
    
    masked_clean = upsampled_clean.cpu()*mask.unsqueeze(0)
    gc.collect()
    masked_dirt = upsampled_dirt.cpu()*mask.unsqueeze(0)
    gc.collect()
    unmasked_original = (upsampled_dirt*(mask==0).unsqueeze(0).to(self.device)).cpu()
    gc.collect()
    
    masked_clean_embedding = torch.sum(masked_clean, dim = [2, 3])/torch.sum(mask != 0)
    masked_dirt_embedding = torch.sum(masked_dirt, dim = [2, 3])/torch.sum(mask != 0)
    unmasked_original_embedding = torch.sum(unmasked_original, dim = [2, 3])/torch.sum(mask == 0)
  
    masked_clean_masked_dirt = cosine_similarity(masked_clean_embedding.detach().numpy(), masked_dirt_embedding.detach().numpy())
    gc.collect()
    masked_clean_unmasked_original = cosine_similarity(masked_clean_embedding.detach().numpy(), unmasked_original_embedding.detach().numpy())
    gc.collect()
    masked_dirt_unmasked_original = cosine_similarity(masked_dirt_embedding.detach().numpy(), unmasked_original_embedding.detach().numpy())
    gc.collect()
  
    return {
              "clean_mask-dirty_mask": float(masked_clean_masked_dirt),
              "clean_mask-original": float(masked_clean_unmasked_original),
              "dirty_mask-original": float(masked_dirt_unmasked_original)
            }
      
  
