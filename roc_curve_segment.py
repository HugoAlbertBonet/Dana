import argparse
import itertools
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import numpy as np


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
from imagedataset import SegmentationDataset



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./segmentation', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=1024, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
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
test_dataloader = DataLoader(SegmentationDataset(opt.dataroot, transforms_=transforms_, transforms_post = transforms_post, unaligned=False, mode = "test"),
                        batch_size=opt.batchSize, shuffle=True)


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

all_preds = []
all_targets = []

model.eval()
with torch.no_grad():
    for j, val_batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)-1):
        if j == len(test_dataloader) - 1:
            break
        val_image = Variable(input_unet.copy_(val_batch['image']))
        val_mask = Variable(input_mask.copy_(val_batch['mask']))

        val_pred = model(val_image)
        val_pred_sigmoid = torch.sigmoid(val_pred)

        all_preds.extend(val_pred_sigmoid.cpu().numpy().flatten())
        all_targets.extend((val_mask.cpu().numpy().flatten()).astype(np.uint8))

print(val_pred_sigmoid.cpu().numpy().flatten().shape, (val_mask.cpu().numpy().flatten()).astype(np.uint8).shape)

# Compute ROC curve and AUC
print("Computing ROC curves...")
fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
roc_auc = auc(fpr, tpr)
print("Thresholds:", thresholds)

# Find the optimal threshold using Youden Index
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
optimal_fpr = fpr[optimal_idx]
optimal_tpr = tpr[optimal_idx]

# Find point where FP = FN (FPR ~= 1 - TPR)
fp_eq_fn_idx = np.argmin(np.abs(fpr - (1 - tpr)))
fp_eq_fn_threshold = thresholds[fp_eq_fn_idx]
fp_eq_fn_fpr = fpr[fp_eq_fn_idx]
fp_eq_fn_tpr = tpr[fp_eq_fn_idx]

print("Plotting ROC curves...")
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')

# Plot dot for Youden Index optimal threshold
plt.plot(optimal_fpr, optimal_tpr, 'ro', label='Youden Index\nThreshold = {:.2f}'.format(optimal_threshold))

# Plot dot for FP = FN
plt.plot(fp_eq_fn_fpr, fp_eq_fn_tpr, 'bo', label='FP = FN\nThreshold = {:.2f}'.format(fp_eq_fn_threshold))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for UNet Segmentation')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("plots/roc_curve.png")
plt.close()



# Plot DICE score vs Threshold
import numpy as np
print("Computing DICE vs threshold...")

thresholds = np.linspace(0, 1, 101)
dice_scores = []

all_preds_np = np.array(all_preds)
all_targets_np = np.array(all_targets)

for thr in tqdm(thresholds):
    pred_bin = (all_preds_np > thr).astype(np.float32)
    mask = all_targets_np.astype(np.float32)
    
    intersection = np.sum(pred_bin * mask)
    dice = (2. * intersection) / (np.sum(pred_bin) + np.sum(mask) + 1e-8)
    dice_scores.append(dice)

print("Plotting DICE vs threshold...")
# Plot the Dice curve
plt.figure()
plt.plot(thresholds, dice_scores, color='green', lw=2)
plt.xlabel('Threshold')
plt.ylabel('DICE Score')
plt.title('DICE Score vs Threshold')
plt.grid(True)
plt.savefig("plots/dice_vs_threshold.png")
plt.close()

print("Done")
"""
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
print(f'Saved at: trials/masked_{new_name}.jpg')"""
