from samgeo.text_sam import LangSAM
from PIL import Image, ImageFilter
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

class FaceExtractor():
  def __init__(self, model_size:["tiny", "small", "base-plus", "large"] = "tiny", device = "cpu"):
    self.sam = LangSAM(model_type=f"sam2-hiera-{model_size}")
    self.model_size = model_size
    self.sam.device = device

  def print_result(self):
    self.sam.show_anns(
          cmap="Greens",
          box_color="red",
          title="Automatic Segmentation of Faces",
          blend=True,
      )

  def __call__(self, image:[str, Image], prompt:str = "face", box_threshold=0.24,text_threshold=0.24, output:[str, None]=None, mask_multiplier:int=1, dtype=np.uint8, return_results:["mask", "box", "both", None]=None, print_result:bool = True):
    #Print results only works when result is not returned
    if isinstance(image, str):
      image = Image.open(image)

    self.sam.predict(image, text_prompt=prompt, box_threshold=box_threshold, text_threshold=text_threshold, output = output, mask_multiplier=mask_multiplier, dtype=dtype)

    if return_results == "mask": return self.sam.prediction.astype(np.uint8)
    elif return_results == "box": return self.sam.boxes.cpu().numpy() #format y1, x1, y2, x2
    elif return_results == "both": return self.sam.prediction.astype(np.uint8), self.sam.boxes.cpu().numpy()

    if print_result:
      self.print_result()
