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
parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./images/', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
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
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

netG_dirt2clean.load_state_dict(torch.load("./output/netG_dirt2clean_swin.pth", weights_only=True))
netD_A.load_state_dict(torch.load("./output/netD_A.pth", weights_only=True))
netD_B.load_state_dict(torch.load("./output/netD_B.pth", weights_only=True))


if opt.cuda:
    netG_dirt2clean.cuda()
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
dataloader = DataLoader(ImageNetDataset(opt.dataroot, transforms_=transforms_, transforms_dirt = transforms_dirt, transforms_post = transforms_post, unaligned=False),
                        batch_size=opt.batchSize, shuffle=True) #, num_workers=opt.n_cpu)
###################################

invTrans = transforms.Normalize(
   mean=[0,0,0],
   std=[1/255, 1/255, 1/255]
)


input_clean = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)

i = 0
for batch in dataloader:
    images = batch
    image = Variable(input_clean.copy_(batch['B']))
    i += 1
    if i == 3: break


for i in range(opt.batchSize):
    fake_A = netG_dirt2clean(image)
    import numpy as np
    print(torch.max(fake_A[i].type(torch.uint8)), torch.min(fake_A[i].type(torch.uint8)))
    
    transforms.functional.to_pil_image(fake_A[i]).save('trials/fake_clean2.jpg')
    transforms.functional.to_pil_image(images["A"][i]).save('trials/real_clean2.jpg')
    transforms.functional.to_pil_image(images["B"][i]).save('trials/dirt2.jpg')