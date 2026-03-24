import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
from pycocotools.coco import COCO

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

# Custom PyTorch Dataset to load COCO-format annotations and images
class FashionDatasetMaskRCNN(Dataset):
    # Init function: loads annotation file and prepares list of image id's
    def __init__(self, root_dir, annotation_file, transforms=None, target_height=768, target_width=512):
        """
        root_dir: path to the folder containing images (e.g. car_parts_dataset/train/)
        annotation_file: path to the COCO annotations (e.g. car_parts_dataset/train/_annotations.coco.json)
        """
        self.root_dir = root_dir  # Directory where images are stored
        self.coco = COCO(annotation_file)  # Load COCO annotations
        self.image_ids = list(self.coco.imgs.keys())  # Extract all image IDs
        self.transforms = transforms  # Optional image transformations
        self.target_height = target_height
        self.target_width = target_width
    # Returns total number of images
    def __len__(self):
        return len(self.image_ids)  # Total number of images
 
    # Fetches a single image and its annotations
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]  # Get image ID from list
        image_info = self.coco.loadImgs(image_id)[0]  # Load image info (e.g. filename)
        image_path = os.path.join(self.root_dir, image_info["file_name"])  # Construct full path
        image = Image.open(image_path).convert("RGB")  # Load and convert image to RGB
        w, h = image.size
        scale = min(self.target_width/w, self.target_height/h) #keep aspect ratio
        new_w = int(w*scale)
        new_h = int(h*scale)
        x_offset =(self.target_width - new_w)//2
        y_offset =(self.target_height - new_h)//2
        #resize keeping aspect ratio
        image =image.resize((new_w, new_h), Image.BILINEAR)
        padded_image =Image.new("RGB", (self.target_width, self.target_height), (0, 0, 0))
        padded_image.paste(image,(x_offset, y_offset))
        
        # Load all annotations for this image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)  # Get annotation IDs for image
        annotations = self.coco.loadAnns(annotation_ids)  # Load annotation details
         
        # Extract segmentation masks, bounding boxes and labels from annotations
        boxes = []  # List to store bounding boxes
        labels = []  # List to store category labels
        masks = []  # List to store segmentation masks
        areas = []
        iscrowd = []
        for ann in annotations:
            if ann["category_id"]>27:continue
            xmin, ymin, w, h = ann['bbox']  # Get bounding box in COCO format (x, y, width, height)
            xmin = xmin * scale + x_offset
            ymin = ymin * scale + y_offset
            xmax = xmin + w * scale
            ymax = ymin + h * scale
            boxes.append([xmin, ymin, xmax, ymax])  # Append box in (xmin, ymin, xmax, ymax) format
            labels.append(ann['category_id'])  # Append category ID
            mask = self.coco.annToMask(ann)  # Convert segmentation to binary mask
            mask = Image.fromarray(mask)
            mask =mask.resize((new_w, new_h), Image.NEAREST)
            padded_mask =Image.new("L", (self.target_width, self.target_height), 0)
            padded_mask.paste(mask,(x_offset, y_offset))
            mask = np.array(padded_mask)
            if mask.ndim == 3:
                mask = mask[:,:,0]
            masks.append(mask)  # Append mask
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))
        if len(boxes) == 0:
            boxes= torch.zeros((0,4),dtype=torch.float32)
            labels= torch.zeros((0,), dtype=torch.int64)
            masks =torch.zeros((0, self.target_height, self.target_width), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            # Convert annotations to PyTorch tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)  # Bounding boxes as float tensors
            labels = torch.as_tensor(labels, dtype=torch.int64)  # Labels as int64 tensors
            masks = torch.as_tensor(masks, dtype=torch.uint8)  # Masks as uint8 tensors
            area = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        # store everything in a dictionary
        target = {
            "boxes": boxes,  # Bounding boxes
            "labels": labels,  # Object labels
            "masks": masks,  # Segmentation masks
            "image_id": image_id,  # Image ID
            "area": area,  # Area of each object
            "iscrowd": iscrowd  # Crowd flags
        }
 
        # Apply transforms
        if self.transforms:
            image = self.transforms(padded_image)
        else:
            image = T.ToTensor()(padded_image) 
        # Return the processed image and its annotations
        return image, target  # Return tuple of image and annotation dictionary