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
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--name', type=str, default="", help='name of file to test')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_dirt2clean = SwinIR(upscale=1, img_size=(opt.size, opt.size),
                   window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
unet_dirt2clean = UNet()
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

netG_dirt2clean.load_state_dict(torch.load("./outputGAN/netG_dirt2clean_best.pth", weights_only=True))
unet_dirt2clean.load_state_dict(torch.load("./output/netG_dirt2clean_colorizing.pth", weights_only=True))
netD_A.load_state_dict(torch.load("./output/netD_A.pth", weights_only=True))
netD_B.load_state_dict(torch.load("./output/netD_B.pth", weights_only=True))


if opt.cuda:
    netG_dirt2clean.cuda()
    unet_dirt2clean.cuda()
    netD_A.cuda()
    netD_B.cuda()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip()]
transforms_dirt = []#transforms.ElasticTransform(alpha=50.0)]
transforms_post = [transforms.ToTensor()]

transform = transforms.Compose(transforms_ + transforms_post)
###################################

name = opt.name
item_dirt = Image.open(name).convert('RGB')
item_dirt = transform(item_dirt)
item_dirt = item_dirt.reshape((1, 3, 128, 128))
item_dirt = item_dirt.to("cuda")



fake_A = netG_dirt2clean(item_dirt)
    
new_name = name.split("/")[-1]
    
transforms.functional.to_pil_image(fake_A[0]).save(f'trials/fakeSwin_{new_name}.jpg')
transforms.functional.to_pil_image(item_dirt[0]).save(f'trials/real_{new_name}.jpg')

fake_A = unet_dirt2clean(item_dirt)
    
    
transforms.functional.to_pil_image(fake_A[0]).save(f'trials/fakeUnet_{new_name}.jpg')