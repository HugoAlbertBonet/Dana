import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from datasets import load_dataset
import requests
import io
import numpy as np
import torch
#ds = load_dataset("YangQiee/HQ-50K")

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=[], transforms_dirt = [], transforms_post = [transforms.ToTensor()], unaligned=False, mode='train', use_hq = False):
        self.transform = transforms.Compose(transforms_ + transforms_post)
        self.transform_dirt = transforms.Compose(transforms_dirt)
        self.transforms_post = transforms.Compose(transforms_post)
        self.mode = mode
        self.use_hq = use_hq
        self.unaligned = unaligned
        self.files_hq = ds["train"]["text"]
        self.files_clean = sorted(glob.glob(os.path.join(root, '%s/clean' % mode) + '/*.*'))
        self.files_rain = sorted(glob.glob(os.path.join(root, '%s/rain' % mode) + '/*.*'))
        self.files_mud = sorted(glob.glob(os.path.join(root, 'mud') + '/*.*'))
        self.files_textures = sorted(glob.glob(os.path.join(root, 'texturas' ) + '/*.*'))
        if mode == "test":
            self.files_mud = self.files_mud[round(0.8*len(self.files_mud)):]
            self.files_textures = self.files_textures[round(0.8*len(self.files_textures)):]
        else:
            self.files_mud = self.files_mud[:round(0.8*len(self.files_mud))]
            self.files_textures = self.files_textures[:round(0.8*len(self.files_textures))]

    def __getitem__(self, index):
        p_mud, p_texture, p_hq = random.random(), random.random(), random.random()
        #print(p_mud, p_texture, p_hq)
        if p_hq < 0.5 or not self.use_hq or self.mode == "test":
            item_clean = self.transform(Image.open(self.files_clean[index % len(self.files_clean)]))
        else:
            try:
                r = requests.get(self.files_hq[random.randint(0, len(self.files_hq) - 1)])
                item_clean = self.transform(Image.open(io.BytesIO(r.content)))
            except: item_clean = self.transform(Image.open(self.files_clean[index % len(self.files_clean)]))

        
        done = False
        while not done:
          try:
            if self.unaligned:
              name = self.files_rain[random.randint(0, len(self.files_rain) - 1)]
            else:
              name = self.files_rain[index % len(self.files_rain)]
            item_rain = self.transform_dirt(Image.open(name))
            done = True
          except:
            print(f"Could not open {name}")

        item_clean2 = item_clean.copy()
        if p_texture < 0.1:pass
        else:
            item_texture = self.transform(Image.open(self.files_textures[random.randint(0, len(self.files_textures) - 1)]))
            item_clean2 = Image.blend(item_clean2.convert('RGBA'), item_texture.resize(item_clean2.size).convert('RGBA'), random.uniform(0.2, 0.6))
        if p_mud < 1:
            item_clean = self.transforms_post(item_clean.convert('RGB'))
            item_clean2 = self.transforms_post(self.transform_dirt(item_clean2.convert('RGB')))
            return {'A': item_clean, 'B': item_clean2}
        
        else:
            alpha = random.uniform(0.6, 0.9)
            done = False
            while not done:
              try:
                name = self.files_mud[random.randint(0, len(self.files_mud) - 1)]
                item_mud = Image.open(name)
                done = True
              except:
                print(f"Could not open {name}")
            item_mud = item_mud.resize(item_clean2.size).convert('RGBA')
            datas = item_mud.getdata()
            newData = []
            for item in datas:
                item = (item[0],item[1], item[2], round(item[-1]*alpha))
                newData.append(item)
            item_mud.putdata(newData)
            item_mud = transforms.RandomAffine(90, translate = (0.3,0.3), scale = (0.5,1.5), shear = 20)(item_mud)
            item_clean2.paste(item_mud.convert('RGBA'), (0, 0), item_mud.convert('RGBA'))
            
            item_clean = self.transforms_post(item_clean.convert('RGB'))
            item_clean2 = self.transforms_post(self.transform_dirt(item_clean2.convert('RGB')))
            return {'A': item_clean, 'B': item_clean2}


    def __len__(self):
        return max(len(self.files_clean), len(self.files_rain))
        
        
        
        
        
        
        
        
class ImageNetDataset(Dataset):
    
    def __init__(self, root, transforms_=[], transforms_dirt = [], transforms_post = [transforms.ToTensor()], unaligned=False, mode='train'):
        self.length = 100000
        self.transform = transforms.Compose(transforms_)
        self.transform_dirt = transforms.Compose(transforms_dirt)
        self.transforms_post = transforms.Compose(transforms_post)
        self.imagenet = load_dataset("imagenet-1k")
        self.mode = mode
        self.unaligned = unaligned
        self.files_mud = sorted(glob.glob(os.path.join(root, 'mud') + '/*.*'))
        self.files_textures = sorted(glob.glob(os.path.join(root, 'texturas' ) + '/*.*'))
        if mode == "test":
            self.files_mud = self.files_mud[round(0.8*len(self.files_mud)):]
            self.files_textures = self.files_textures[round(0.8*len(self.files_textures)):]
        else:
            self.files_mud = self.files_mud[:round(0.8*len(self.files_mud))]
            self.files_textures = self.files_textures[:round(0.8*len(self.files_textures))]

    def __getitem__(self, index):
        p_mud, p_texture, p_hq = random.random(), random.random(), random.random()
        #print(p_mud, p_texture, p_hq)
        item_clean = self.transform(self.imagenet[self.mode][index % self.length]["image"]) #len(self.imagenet[self.mode])

        item_clean2 = item_clean.copy()
        if p_texture < 0.1:pass
        else:
            item_texture = self.transform(Image.open(self.files_textures[random.randint(0, len(self.files_textures) - 1)]))
            item_clean2 = Image.blend(item_clean2.convert('RGBA'), item_texture.resize(item_clean2.size).convert('RGBA'), random.uniform(0.2, 0.6))
        if p_mud < 1:
            item_clean = self.transforms_post(item_clean.convert('RGB'))
            item_clean2 = self.transforms_post(self.transform_dirt(item_clean2.convert('RGB')))
            return {'A': item_clean, 'B': item_clean2}
        
        else:
            alpha = random.uniform(0.6, 0.9)
            done = False
            while not done:
              try:
                name = self.files_mud[random.randint(0, len(self.files_mud) - 1)]
                item_mud = Image.open(name)
                done = True
              except:
                print(f"Could not open {name}")
            item_mud = item_mud.resize(item_clean2.size).convert('RGBA')
            datas = item_mud.getdata()
            newData = []
            for item in datas:
                item = (item[0],item[1], item[2], round(item[-1]*alpha))
                newData.append(item)
            item_mud.putdata(newData)
            item_mud = transforms.RandomAffine(90, translate = (0.3,0.3), scale = (0.5,1.5), shear = 20)(item_mud)
            item_clean2.paste(item_mud.convert('RGBA'), (0, 0), item_mud.convert('RGBA'))
            
            item_clean = self.transforms_post(item_clean.convert('RGB'))
            item_clean2 = self.transforms_post(self.transform_dirt(item_clean2.convert('RGB')))
            return {'A': item_clean, 'B': item_clean2}


    def __len__(self):
        return self.length #len(self.imagenet[self.mode])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class ImageNetDanaDataset(Dataset):
    
    def __init__(self, root, transforms_=[], transforms_post = [transforms.ToTensor()], unaligned=False, mode='train'):
        self.length = 100000
        self.transform = transforms.Compose(transforms_ + transforms_post)
        self.transforms_post = transforms.Compose(transforms_post)
        self.imagenet = load_dataset("imagenet-1k")
        self.mode = mode
        self.unaligned = unaligned
        self.files_mud = sorted(glob.glob(root + '/*.*'))
        print(len(self.files_mud))

    def __getitem__(self, index):
        item_clean = self.transform(self.imagenet[self.mode][random.randint(0, len(self.imagenet[self.mode])-1)]["image"]) #len(self.imagenet[self.mode])
        name = self.files_mud[index % len(self.files_mud)]
        item_dirt = Image.open(name).convert('RGB')
        #print(transforms.ToTensor()(item_dirt).shape, item_clean.shape)
        item_dirt = self.transform(item_dirt)
        return {'dirt': item_dirt, 'clean': item_clean}


    def __len__(self):
        return len(self.files_mud)
        
        
        


        
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        
class SegmentationDataset(Dataset):
    
    def __init__(self, root, transforms_=[], transforms_color = [], transforms_post = [transforms.ToTensor()], unaligned=False, mode='train', yolo = False, segformer = False, token = None):
        self.transform = transforms.Compose(transforms_)
        self.transforms_post = transforms.Compose(transforms_post)
        self.transforms_color = transforms.Compose(transforms_color + transforms_post)
        self.images = sorted(glob.glob(root + '/images/*.*'))
        self.masks = [mask for mask in sorted(glob.glob(root + '/masks/*.*')) if "0006-MG005.png" not in mask]
        self.masks_retrain = sorted(glob.glob(root + '/retrain/masks/*.*'))
        self.all_masks = self.masks + self.masks_retrain
        if mode == "test":
            self.masks = self.masks[round(0.8*len(self.masks)):]
        else:
            self.masks = self.masks[:round(0.8*len(self.masks))] + self.masks_retrain
        self.mode = mode
        self.unaligned = unaligned
        self.yolo = yolo
        self.segformer = segformer
        if segformer:
          self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512", token = token)

    def __getitem__(self, index):
        name = self.masks[index]
        #print(name)
        #print(name[:-3].replace("Masks", "Images")+"jpg")
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img transforms
        torch.manual_seed(seed)
        image = self.transform(Image.open(name[:-3].replace("masks", "images")+"jpg").convert('RGB') )
        random.seed(seed) # apply this seed to mask transforms
        torch.manual_seed(seed)
        mask = self.transforms_post(self.transform(Image.open(name).convert('L')))
        peoplemask = self.transforms_post(self.transform(Image.open(name.replace("masks", "peoplemask")).convert('L')))
        if self.yolo:
          yolomask = self.transforms_post(self.transform(Image.open(name.replace("masks", "yolomask")).convert('L')))
        image = self.transforms_color(image)
        if self.segformer:
          image = self.processor(images=transforms.functional.to_pil_image(image).resize((512, 512)), return_tensors="pt")
        if self.yolo:
          return {'image': image, 'mask': mask, "people": peoplemask, "yolo": yolomask}
        else:
          return {'image': image, 'mask': mask, "people": peoplemask}

    def __len__(self):
        return len(self.masks)
    
    
    
    
    

if __name__ == "__main__":
    """dataset = ImageDataset("./images/", mode = "test", transforms_= [], transforms_post= [transforms.ToTensor()])
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.pyplot as plt
    fig = plt.figure(1,(10,10))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(2,2),
                    axes_pad=0.1,
                    )
    images = dataset[0]
    grid[0].imshow(transforms.functional.to_pil_image(images["A"]),cmap='gray',interpolation='none')
    grid[0].axis("off")
    grid[0].set_title("Clean")
    grid[1].imshow(transforms.functional.to_pil_image(images["B"]),cmap='gray',interpolation='none')
    grid[1].axis("off")
    grid[1].set_title("Not Clean")
    images = dataset[1]
    grid[2].imshow(transforms.functional.to_pil_image(images["A"]),cmap='gray',interpolation='none')
    grid[2].axis("off")
    grid[2].set_title("")
    grid[3].imshow(transforms.functional.to_pil_image(images["B"]),cmap='gray',interpolation='none')
    grid[3].axis("off")
    grid[3].set_title("")
    plt.show()  # Display the plot"""
    
    ImageNetDataset()
    
    
    
    
    
    
    
    
    
    
