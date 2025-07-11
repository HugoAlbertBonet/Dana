from segformer import SegformerInference
from metrics import reMOVE
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

device = "cuda:1"

segformer = SegformerInference(device)
remove = reMOVE(device = device, checkpoint = "/home/salvem/hugoa/gradio/libs/sam_vit_b_01ec64.pth")

all_images = glob.glob("/home/salvem/jgalvan/restauraciones/yolo8_10_17_unet_segformer/*")

images = glob.glob("/home/salvem/benchmark_v2_eval/originales/*")

d = {"model":[], "image":[], "Prev_DP":[], "New_DP":[], "DPR":[], "RC_cm_dm":[], "RC_cm_og":[], "RC_dm_og":[], "RCR": []}

"""for x in images:
  if x.replace("benchmark_v2_eval/originales/", "jgalvan/restauraciones/yolo8_10_17_unet_segformer/").replace(".jpg", f"_MASK_ORIGINAL_SegFormer.png") not in all_images:
    print(x)"""

for x in tqdm(images):
  name = x.split("/")[-1]
  if x.replace("benchmark_v2_eval/originales/", "jgalvan/restauraciones/yolo8_10_17_unet_segformer/").replace(".jpg", f"_MASK_ORIGINAL_SegFormer.png") in all_images:
    for model in ["SegFormer", "UNet", "YOLO+SAM"]:
      d["model"].append(model)
      d["image"].append(name)
      new_root = x.replace("benchmark_v2_eval/originales/", "jgalvan/restauraciones/yolo8_10_17_unet_segformer/")
      mask_inpainting = new_root.replace(".jpg", f"_MASK_REFINED_{model}.png")
      mask_segmentation = new_root.replace(".jpg", f"_MASK_ORIGINAL_SegFormer.png")
      restored_image = new_root.replace(".jpg", f"_RESTORED_{model}.png")
      
      mask_array = np.array(Image.open(mask_segmentation).convert("L"))/255
      h, w = mask_array.shape
      prev_dp = np.sum(mask_array)/(h*w)
      d["Prev_DP"].append(prev_dp)
      
      image_array = np.array(Image.open(restored_image))
      mask_restored = segformer.get_mask(image_array)[:,:,0].squeeze()/255
      h, w = mask_restored.shape
      new_dp = np.sum(mask_restored)/(h*w)
      d["New_DP"].append(new_dp)
      d["DPR"].append(new_dp/prev_dp)
      
      remove_metrics = remove(mask_inpainting, x, restored_image)
      
      d["RC_cm_dm"].append(remove_metrics["clean_mask-dirty_mask"])
      d["RC_cm_og"].append(remove_metrics["clean_mask-original"])
      d["RC_dm_og"].append(remove_metrics["dirty_mask-original"])
      d["RCR"].append(remove_metrics["clean_mask-original"]/remove_metrics["dirty_mask-original"])

df = pd.DataFrame(d)
df.to_csv("metrics.csv")
df.to_excel("metrics.xlsx")

