import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np

class FashionDataset(Dataset):
    def __init__(self, img_dir, mask_dir, target_height=768, target_width=512, transform=None, originals=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_height = target_height
        self.target_width = target_width
        mask_basenames = [basename.split("_")[0] for basename in os.listdir(mask_dir)]
        basenames = [base.split(".")[0] for base in os.listdir(img_dir) if base.split(".")[0] in mask_basenames]
        name_img_files = [basename +'.jpg' for basename in basenames]
        mask_img_files = [basename +'_seg.png' for basename in basenames]
        self.img_files = sorted(name_img_files)
        self.mask_files = sorted(mask_img_files)
        self.originals = originals

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image_orig = Image.open(img_path).convert('RGB')
        mask_orig = Image.open(mask_path)

        w, h = image_orig.size
        scale = min(self.target_width/w, self.target_height/h) #keep aspect ratio
        new_w = int(w*scale)
        new_h = int(h*scale)

        #resize keeping aspect ratio
        image =image_orig.resize((new_w, new_h), Image.BILINEAR)
        mask =mask_orig.resize((new_w, new_h), Image.NEAREST)
        #create empty images
        padded_image =Image.new("RGB", (self.target_width, self.target_height), (0, 0, 0))
        padded_mask =Image.new("L", (self.target_width, self.target_height), 0)
        #center position
        x_offset =(self.target_width - new_w)//2
        y_offset =(self.target_height - new_h)//2
        padded_image.paste(image,(x_offset, y_offset))
        padded_mask.paste(mask,(x_offset, y_offset))
        
        mask = np.array(padded_mask)
        if mask.ndim == 3:
            mask = mask[:,:,0]
        if self.transform:
            image = self.transform(padded_image)
        else:
            image = T.ToTensor()(padded_image)
        mask = torch.as_tensor(mask, dtype=torch.long)
        if self.originals:
            return image_orig, mask_orig, image, mask
        return image, mask
