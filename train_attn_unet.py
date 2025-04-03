import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from x_unet import XUnet

from models import Generator, UNet
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from imagedataset import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=10, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./images/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_dirt2clean = XUnet(
    dim = 64,
    channels = 3,
    dim_mults = (1, 2, 4, 8),
    nested_unet_depths = (7, 4, 2, 1),     # nested unet depths, from unet-squared paper
    consolidate_upsample_fmaps = True,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
)
netD_clean = Discriminator(opt.input_nc)


if opt.cuda:
    netG_dirt2clean.cuda()
    netD_clean.cuda()

netD_clean.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.MSELoss()
criterion_identity = torch.nn.MSELoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(netG_dirt2clean.parameters(),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD_clean.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_clean = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_dirt = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_clean_buffer = ReplayBuffer()
fake_dirt_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip()]
transforms_dirt = [transforms.ElasticTransform(alpha=50.0)]
transforms_post = [transforms.ToTensor()]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, transforms_dirt = transforms_dirt, transforms_post = transforms_post, unaligned=False),
                        batch_size=opt.batchSize, shuffle=True) #, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_clean = Variable(input_clean.copy_(batch['A']))
        real_dirt = Variable(input_dirt.copy_(batch['B']))

        ###### Generators dirt2clean ######
        optimizer_G.zero_grad()

        # GAN loss
        fake_clean = netG_dirt2clean(real_dirt)
        pred_fake = netD_clean(fake_clean)
        loss_GAN = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        loss_cycle = criterion_cycle(fake_clean, real_clean)*10.0

        # Total loss
        loss_G = loss_GAN + loss_cycle
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D.zero_grad()

        # Real loss
        pred_real = netD_clean(real_clean)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_clean = fake_clean_buffer.push_and_pop(fake_clean)
        pred_fake = netD_clean(fake_clean.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D = (loss_D_real + loss_D_fake)*0.5
        loss_D.backward()

        optimizer_D.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_GAN': loss_GAN,
                    'loss_G_cycle': loss_cycle, 'loss_D': loss_D},
                    images={'real_A': real_clean, 'real_B': real_dirt, 'fake_A': fake_clean, 'fake_B': None})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    # Save models checkpoints
    torch.save(netG_dirt2clean.state_dict(), 'output/attnUnet_dirt2clean.pth')
    torch.save(netD_clean.state_dict(), 'output/netD_clean.pth')