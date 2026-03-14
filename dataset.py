import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np

class FashionDataset(Dataset):
    def __init__(self, img_dir, mask_dir, target_height=768, target_width=512, transform=None):
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

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        image = image.resize((self.target_width, self.target_height), Image.BILINEAR)
        mask = mask.resize((self.target_width, self.target_height), Image.NEAREST)
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[:,:,0]
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        mask = torch.as_tensor(mask, dtype=torch.long)
        return image, mask

