import os
import config
import cv2
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

images=os.listdir(config.TRAIN_IMG)
masks=os.listdir(config.TRAIN_MASK)
fabrics = os.listdir(config.FABRICS)
os.makedirs(os.path.join(config.ROOT, "generated"), exist_ok=True)
no_possible_ids = [5, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27]
ph, pw = 15, 15
wanted= int(len(images)*0.05)
selected = random.choices(images, k=wanted)
for j,path_img in enumerate(selected):
    mask_path = path_img.split(".")[0] + "_seg.png"
    if mask_path not in masks: continue
    img= Image.open(os.path.join(config.TRAIN_IMG, path_img)).convert("RGB")
    img_orig = np.array(img)
    img_result = np.array(img).copy()
    mask = Image.open(os.path.join(config.TRAIN_MASK, mask_path)).convert("L")
    mask_np =np.array(mask)
    uniques = np.unique(mask_np)
    uniques = uniques[uniques!=0]
    filtered_uniques = [unique for unique in uniques if unique not in no_possible_ids]
    if len(filtered_uniques)==0: continue
    selected_fabrics = random.choices(fabrics,k=1)
    selected_item =random.choices(filtered_uniques, k=1)
    h,w = mask_np.shape
    mask_selected = (mask_np ==selected_item[0])
    fabric = Image.open(os.path.join(config.FABRICS, selected_fabrics[0])).convert("RGB")
    fabric_patch = cv2.resize(np.array(fabric), (30,30))
    tile_y=int(np.ceil(h/ph))
    tile_x=int(np.ceil(w/pw))
    fabric_resized=np.tile(fabric_patch, (tile_y, tile_x,1))
    fabric_resized=fabric_resized[:h, :w, :]
    img_result[mask_selected] = fabric_resized[mask_selected]
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(img_orig)
    axs[0].set_title("original")
    axs[0].axis("off")
    axs[1].imshow(img_result)
    axs[1].set_title("Generated")
    axs[1].axis("off")
    plt.savefig(f"{config.ROOT}/generated/generated_{j}.png")
    plt.close(fig)