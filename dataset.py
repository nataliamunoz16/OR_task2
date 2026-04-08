import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
from pycocotools.coco import COCO

class FashionDataset(Dataset):
    def __init__(self, img_dir, mask_dir, target_height=768, target_width=512, transform=None, originals=False, overrepresented_ids=[], crops_minor=False):
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
        self.crops_minor=crops_minor
        self.overrepresented_ids= overrepresented_ids

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
        if len(self.overrepresented_ids)>0 and (self.crops_minor == False):
            mask[np.isin(mask, self.overrepresented_ids)]=255
            #valid= (mask!=255).sum().item()
            #if valid == 0:
            #    print(f"Sample {idx} has no valid pixels after filtering")
        if self.transform:
            image = self.transform(padded_image)
        else:
            image = T.ToTensor()(padded_image)
        mask = torch.as_tensor(mask, dtype=torch.long)
        if self.originals:
            return image_orig, mask_orig, image, mask
        return image, mask

class FashionDatasetCropped(Dataset):
    def __init__(self, img_dir, mask_dir, annotations_json, target_height=768, target_width=512, transform=None, originals=False, overrepresented_ids=[]):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_height = target_height
        self.target_width = target_width
        self.originals = originals
        self.overrepresented_ids= overrepresented_ids
        self.coco = COCO(annotations_json)
        available_imgs=set(os.listdir(img_dir))
        available_masks=set(os.listdir(mask_dir))

        self.samples=[]
        for img_id in self.coco.getImgIds():
            img_info=self.coco.imgs[img_id]
            img_file=img_info["file_name"]
            mask_file = os.path.splitext(img_file)[0]+"_seg.png"
            if (img_file not in available_imgs) or (mask_file not in available_masks): continue

            self.samples.append({"type": "full", "img_file": img_file, "mask_file": mask_file, "bbox": None})
            annotation_ids=self.coco.getAnnIds(imgIds=img_id)
            annotations=self.coco.loadAnns(annotation_ids)
            for ann in annotations:
                if ((ann["category_id"]+1)<28) and ((ann["category_id"]+1) not in self.overrepresented_ids):
                    self.samples.append({"type": "crop", "img_file": img_file, "mask_file": mask_file, "bbox": ann["bbox"]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample["img_file"])
        mask_path = os.path.join(self.mask_dir, sample["mask_file"])
        image_orig = Image.open(img_path).convert('RGB')
        mask_orig = Image.open(mask_path)

        if sample["type"]=="crop":
            x, y,w_box, h_box = sample["bbox"]
            if x>0 and y>0 and w_box>0 and h_box>0:
                image_orig = image_orig.crop((int(x), int(y), int(x+w_box), int(y+h_box)))
                mask_orig = mask_orig.crop((int(x), int(y), int(x+w_box), int(y+h_box)))

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