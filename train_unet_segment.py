import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import wandb
import json


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

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
with open("secret.json", "r") as f:
  secret = json.load(f)
wandb.login(key = secret["wandb"])
config = {"learning_rate": opt.lr, 
                "epochs": opt.n_epochs, 
                "batch_size": opt.batchSize,
                "model_name": opt.name,
                "gradient_accumulation_steps": opt.accumsteps, 
                "decay_epoch": opt.decay_epoch,
                "image_size": opt.size,
                "cuda": opt.cuda,
                "description": opt.description}
wandb.init(project="dana-segmentation", config = config)


###### Definition of variables ######
# Networks
model = UNet(n_class = 1)


if opt.cuda:
    model.cuda()

# Magic
wandb.watch(model, log_freq=100)
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
transforms_ = [ transforms.Resize((opt.size, opt.size)), transforms.RandomAffine(30, translate= (0.2,0.2)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGrayscale(p=0.1), transforms.ElasticTransform(alpha=25.0)]
transforms_post = [transforms.ToTensor()]
dataloader = DataLoader(SegmentationDataset(opt.dataroot, transforms_=transforms_, transforms_post = transforms_post, unaligned=False),
                        batch_size=opt.batchSize, shuffle=True) #, num_workers=opt.n_cpu)
test_dataloader = DataLoader(SegmentationDataset(opt.dataroot, transforms_=[transforms.Resize((opt.size, opt.size))], transforms_post = transforms_post, unaligned=False, mode = "test"),
                        batch_size=opt.batchSize, shuffle=True)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader), len(test_dataloader))

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
optimizer.zero_grad()
accum_steps = 0

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    model.train()
    total_loss = 0
    total_dice = 0
    for i, batch in enumerate(dataloader):
        if i == len(dataloader)-1: break
        # Set model input
        image = Variable(input_unet.copy_(batch['image']))
        mask = Variable(input_unet2.copy_(batch['mask']))

        ###### main loop ######

        #loss
        pred = model(image)
        loss = criterion(pred, mask)
        total_loss += loss.item()
        train_dice = dice(pred, mask)
        total_dice += train_dice
        loss.backward()
        
        #gradient accumulation
        if accum_steps == opt.accumsteps:
          optimizer.step()
          optimizer.zero_grad()
        #print(pred.shape, mask.shape)
        
        if accum_steps == opt.accumsteps: accum_steps = 0
        accum_steps += 1

        # Progress report (http://localhost:8097)
        logger.log({'loss': loss, 'DICE': train_dice})
        wandb.log({'loss': loss, 'DICE': train_dice})
        
        # Validation phase
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for j, val_batch in enumerate(test_dataloader):
            if j == len(test_dataloader)-1: break
            val_image = Variable(input_unet.copy_(val_batch['image']))
            val_mask = Variable(input_unet2.copy_(val_batch['mask']))

            val_pred = model(val_image)
            batch_loss = criterion(val_pred, val_mask)
            batch_dice = dice(val_pred, val_mask)

            val_loss += batch_loss.item()
            val_dice += batch_dice.item()
            #logger.log({'val_loss': batch_loss, 'val_DICE': batch_dice})

    # Average validation loss and DICE
    val_loss /= len(test_dataloader)
    val_dice /= len(test_dataloader)

    print(f"\nEpoch {epoch+1}/{opt.n_epochs} - Validation Loss: {val_loss:.4f}, DICE: {val_dice:.4f}")
    wandb.log({'train_loss': total_loss/len(dataloader), 'train_DICE': total_dice/len(dataloader), 'val_loss': val_loss, 'val_DICE': val_dice, "epoch": epoch, "lr": lr_scheduler.get_last_lr()[-1]})
    #torch.onnx.export(model, val_image, "model.onnx")
    #wandb.save("model.onnx")
    

    # Update learning rates
    lr_scheduler.step()

    # Save models checkpoints
    torch.save(model.state_dict(), f'output/{opt.name}.pth')
