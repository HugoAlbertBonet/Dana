import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import wandb
import json
import numpy as np
import glob
import cv2
from tqdm import tqdm
import gc


from models import Generator, UNet, Colorizing, SwinIR
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal, generate_binary_mask
from imagedataset import SegmentationDataset
from face_extractor import FaceExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=10, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./segmentation', help='root directory of the dataset')
parser.add_argument('--name', type=str, default='', help='description of the experiment for wandb')
parser.add_argument('--description', type=str, default='unet_segmentation', help='name of the saved model')
parser.add_argument('--accumsteps', type=int, default=8, help='gradient accumulation steps')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=1024, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

device = "cuda:1"

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    

detector = FaceExtractor(device = device)


# Dataset loader
transforms_ = []
transforms_color = []
transforms_post = [transforms.ToTensor()]
dataset = SegmentationDataset(opt.dataroot, transforms_=transforms_, transforms_color = transforms_color, transforms_post = transforms_post, unaligned=False)

for im in tqdm(dataset.all_masks):
  image = Image.open(im).convert("RGB")
  mask_people = detector(image, prompt = "people", return_results = "mask", mask_multiplier = 255)
  mask_people = Image.fromarray(mask_people.astype(np.uint8))
  mask_people.save(im.replace("masks", "peoplemask"))
