import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import matplotlib.pyplot as plt
from x_unet import XUnet

from models import Generator, UNet, Colorizing, SwinIR
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from imagedataset import ImageNetDataset



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./images/', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=1024, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--name', type=str, default="", help='name of file to test')
parser.add_argument('--thr', type=float, default=0.5, help='name of file to test')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
print("Loading model...")
model = UNet(n_class = 1)
model.load_state_dict(torch.load("./output/unet_segmentation400epochs_2.pth", weights_only=True))


if opt.cuda:
    model.cuda()


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_unet = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_mask = Tensor(opt.batchSize, 1, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.Resize((opt.size, opt.size))]
transforms_post = [transforms.ToTensor()]

transform = transforms.Compose(transforms_ + transforms_post)
###################################
print(f"Loading image {opt.name}")
name = opt.name
image = Image.open(name).convert('RGB')
image_tensor = transform(image).reshape((1, opt.input_nc, opt.size, opt.size))
if opt.cuda:
  image_tensor = image_tensor.to("cuda")


print("Masking image...")
mask = model(image_tensor)
mask = torch.sigmoid(mask)
mask_bin = (mask > opt.thr).float()
    
new_name = name.split("/")[-1]

print("Saving result...")
masked_image = Image.blend(transforms.functional.to_pil_image(image_tensor[0]),transforms.functional.to_pil_image(mask_bin[0]).convert("RGB"), 0.7)
    
masked_image.save(f'trials/masked_{new_name}.jpg')
image.save(f'trials/real_{new_name}.jpg')
print(f'Saved at: trials/masked_{new_name}.jpg')
