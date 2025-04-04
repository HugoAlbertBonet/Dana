import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator, UNet, Colorizing, SwinIR
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from imagedataset import SegmentationDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=10, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./segmentation', help='root directory of the dataset')
parser.add_argument('--name', type=str, default='unet_segmentation', help='name of the saved model')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=1024, help='size of the data crop (squared assumed)')
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
model = UNet(n_class = 1)


if opt.cuda:
    model.cuda()

#model.apply(weights_init_normal)

# Lossess
criterion = torch.nn.BCEWithLogitsLoss()

# Optimizers & LR schedulers
optimizer = torch.optim.Adam(model.parameters(),
                                lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_unet = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_unet2 = Tensor(opt.batchSize, 1, opt.size, opt.size)


# Dataset loader
transforms_ = [ transforms.Resize((opt.size, opt.size)), transforms.RandomAffine(30, translate= (0.2,0.2)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
transforms_post = [transforms.ToTensor()]
dataloader = DataLoader(SegmentationDataset(opt.dataroot, transforms_=transforms_, transforms_post = transforms_post, unaligned=False),
                        batch_size=opt.batchSize, shuffle=True) #, num_workers=opt.n_cpu)
test_dataloader = DataLoader(SegmentationDataset(opt.dataroot, transforms_=transforms_, transforms_post = transforms_post, unaligned=False, mode = "test"),
                        batch_size=opt.batchSize, shuffle=True)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))

def dice(pred, mask, threshold=0.5):
    # Apply sigmoid to get probabilities if needed
    pred = torch.sigmoid(pred)
    # Binarize predictions using the threshold
    pred_bin = (pred > threshold).float()
    # Flatten the tensors
    pred_bin = pred_bin.view(-1)
    mask = mask.view(-1)
    # Calculate intersection and union
    intersection = (pred_bin * mask).sum()
    dice_score = (2. * intersection) / (pred_bin.sum() + mask.sum() + 1e-8)
    return dice_score
###################################
print("Starting training...")

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    model.train()
    for i, batch in enumerate(dataloader):
        if i == len(dataloader)-1: break
        # Set model input
        image = Variable(input_unet.copy_(batch['image']))
        mask = Variable(input_unet2.copy_(batch['mask']))

        ###### main loop ######
        optimizer.zero_grad()

        #loss
        pred = model(image)
        loss = criterion(pred, mask)
        loss.backward()

        optimizer.step()
        #print(pred.shape, mask.shape)

        # Progress report (http://localhost:8097)
        logger.log({'loss': loss, 'DICE': dice(pred, mask)})
        
    model.eval()
    with torch.no_grad():
      for i, batch in enumerate(test_dataloader):
          if i == len(test_dataloader)-1: break
          # Set model input
          image = Variable(input_unet.copy_(batch['image']))
          mask = Variable(input_unet2.copy_(batch['mask']))
  
          
          pred = model(image)
          loss_val = criterion(pred, mask)
  
          # Progress report (http://localhost:8097)
          logger.log({'loss_val': loss_val, 'DICE_val': dice(pred, mask)})

    # Update learning rates
    lr_scheduler.step()

    # Save models checkpoints
    torch.save(model.state_dict(), f'output/{opt.name}.pth')