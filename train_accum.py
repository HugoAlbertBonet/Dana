

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator, UNet, SwinIR
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from imagedataset import ImageNetDanaDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--accumsteps', type=int, default=8, help='gradient accumulation steps')
parser.add_argument('--dataroot', type=str, default='/home/salvem/fotosIA', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
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
netG_dirt2clean = SwinIR(upscale=1, img_size=(opt.size, opt.size),
                   window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
netG_clean2dirt = SwinIR(upscale=1, img_size=(opt.size, opt.size),
                   window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
netD_dirt = Discriminator(opt.input_nc)
netD_clean = Discriminator(opt.output_nc)
best_loss = 100000000

if opt.epoch > 0:
  netG_dirt2clean.load_state_dict(torch.load('./outputGAN/netG_dirt2clean.pth', weights_only=True))
  netG_clean2dirt.load_state_dict(torch.load('./outputGAN/netG_clean2dirt.pth', weights_only=True))
  netD_dirt.load_state_dict(torch.load('./outputGAN/netD_dirt.pth', weights_only=True))
  netD_clean.load_state_dict(torch.load('./outputGAN/netD_clean.pth', weights_only=True))
  print("Models loaded")

if opt.cuda:
    netG_dirt2clean.cuda()
    netG_clean2dirt.cuda()
    netD_dirt.cuda()
    netD_clean.cuda()

if opt.epoch > 0:
  netG_dirt2clean.apply(weights_init_normal)
  netG_clean2dirt.apply(weights_init_normal)
  netD_dirt.apply(weights_init_normal)
  netD_clean.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_dirt2clean.parameters(), netG_clean2dirt.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_dirt = torch.optim.Adam(netD_dirt.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_clean = torch.optim.Adam(netD_clean.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_dirt = torch.optim.lr_scheduler.LambdaLR(optimizer_D_dirt, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_clean = torch.optim.lr_scheduler.LambdaLR(optimizer_D_clean, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_dirt = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_clean = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_dirt_buffer = ReplayBuffer()
fake_clean_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip() ]
transforms_post = [transforms.ToTensor()]

def process_image(x):
    #a = x.shape
    if x.shape[0] == 3: pass
    else: 
      x = x.repeat(3, 1, 1)
    #print(x.shape, a)
    return x

def collate_fn(batch):
  return {
      'dirt': torch.stack([process_image(x['dirt']) for x in batch]),
      'clean': torch.stack([process_image(x['clean']) for x in batch])
}

dataloader = DataLoader(ImageNetDanaDataset(opt.dataroot, transforms_=transforms_, transforms_post = transforms_post, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, collate_fn=collate_fn)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################
optimizer_G.zero_grad()
optimizer_D_dirt.zero_grad()
optimizer_D_clean.zero_grad()
accum_steps = 0

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_dirt = Variable(input_dirt.copy_(batch['dirt']))
        
        try:
          real_clean = Variable(input_clean.copy_(batch['clean']))
        except:
          continue

        ###### Generators A2B and B2A ######
        

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_clean = netG_dirt2clean(real_clean)
        loss_identity_clean = criterion_identity(same_clean, real_clean)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_dirt = netG_clean2dirt(real_dirt)
        loss_identity_dirt = criterion_identity(same_dirt, real_dirt)*5.0

        # GAN loss
        fake_clean = netG_dirt2clean(real_dirt)
        pred_fake = netD_clean(fake_clean)
        loss_GAN_dirt2clean = criterion_GAN(pred_fake, target_real)

        fake_dirt = netG_clean2dirt(real_clean)
        pred_fake = netD_dirt(fake_dirt)
        loss_GAN_clean2dirt = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_dirt = netG_clean2dirt(fake_clean)
        loss_cycle_dirtcleandirt = criterion_cycle(recovered_dirt, real_dirt)*10.0

        recovered_clean = netG_dirt2clean(fake_dirt)
        loss_cycle_cleandirtclean = criterion_cycle(recovered_clean, real_clean)*10.0

        # Total loss
        loss_G = loss_identity_dirt + loss_identity_clean + loss_GAN_dirt2clean + loss_GAN_clean2dirt + loss_cycle_dirtcleandirt + loss_cycle_cleandirtclean
        loss_G.backward()

        if accum_steps == opt.accumsteps:
          optimizer_G.step()
          optimizer_G.zero_grad()
        ###################################

        ###### Discriminator A ######
        

        # Real loss
        pred_real = netD_dirt(real_dirt)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_dirt = fake_dirt_buffer.push_and_pop(fake_dirt)
        pred_fake = netD_dirt(fake_dirt.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_dirt = (loss_D_real + loss_D_fake)*0.5
        loss_D_dirt.backward()

        if accum_steps == opt.accumsteps:
          optimizer_D_dirt.step()
          optimizer_D_dirt.zero_grad()
        ###################################

        ###### Discriminator B ######
        

        # Real loss
        pred_real = netD_clean(real_clean)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_clean = fake_clean_buffer.push_and_pop(fake_clean)
        pred_fake = netD_clean(fake_clean.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_clean = (loss_D_real + loss_D_fake)*0.5
        loss_D_clean.backward()

        if accum_steps == opt.accumsteps:
          optimizer_D_clean.step()
          optimizer_D_clean.zero_grad()
        ###################################
        if accum_steps == opt.accumsteps: accum_steps = 0
        accum_steps += 1

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_dirt + loss_identity_clean), 'loss_G_GAN': (loss_GAN_dirt2clean + loss_GAN_clean2dirt),
                    'loss_G_cycle': (loss_cycle_dirtcleandirt + loss_cycle_cleandirtclean), 'loss_D': (loss_D_dirt + loss_D_clean)},
                    images={'real_dirt': real_dirt, 'real_clean': real_clean, 'fake_dirt': fake_dirt, 'fake_clean': fake_clean})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_dirt.step()
    lr_scheduler_D_dirt.step()

    # Save models checkpoints
    torch.save(netG_dirt2clean.state_dict(), './outputGAN/netG_dirt2clean_last.pth')
    torch.save(netG_clean2dirt.state_dict(), './outputGAN/netG_clean2dirt_last.pth')
    torch.save(netD_dirt.state_dict(), './outputGAN/netD_dirt_last.pth')
    torch.save(netD_clean.state_dict(), './outputGAN/netD_clean_last.pth')
    
    if loss_G < best_loss:
      torch.save(netG_dirt2clean.state_dict(), './outputGAN/netG_dirt2clean_best.pth')
      torch.save(netG_clean2dirt.state_dict(), './outputGAN/netG_clean2dirt_best.pth')
      torch.save(netD_dirt.state_dict(), './outputGAN/netD_dirt_best.pth')
      torch.save(netD_clean.state_dict(), './outputGAN/netD_clean_best.pth')
      
###################################