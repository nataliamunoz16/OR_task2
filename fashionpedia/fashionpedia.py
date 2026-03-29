import sys
import os
import numpy as np
import cv2
from pycocotools.coco import COCO

file = sys.argv[1]
dest = sys.argv[2]

os.makedirs(dest, exist_ok=True)

db = COCO(file)
im_ids = db.getImgIds()

for im_id in im_ids:
    ann_ids = np.array(db.getAnnIds(imgIds=im_id))

    # Keep only annotations with label <= 27
    valid_ann_ids = []
    cat_ids = []
    for i in ann_ids:
        label = db.anns[i]['category_id'] + 1
        if label <= 27:
            valid_ann_ids.append(i)
            cat_ids.append(label)

    valid_ann_ids = np.array(valid_ann_ids)

    # Sort by label
    if len(valid_ann_ids) > 0:
        valid_ann_ids = valid_ann_ids[np.argsort(cat_ids)]

    mask = None
    for i in valid_ann_ids:
        label = db.anns[i]['category_id'] + 1
        m = label * db.annToMask(db.anns[i])
        mm = (m > 0).astype(np.uint8)

        if mask is None:
            mask = m
        else:
            mask = mask * (1 - mm) + m

    # If no valid labels were found, create an empty mask
    if mask is None:
        h = db.imgs[im_id]['height']
        w = db.imgs[im_id]['width']
        mask = np.zeros((h, w), dtype=np.uint8)

    name = db.imgs[im_id]['file_name'].split('.')[0] + '_seg.png'
    cv2.imwrite(os.path.join(dest, name), mask.astype(np.uint8))
    print('file ' + name + ' has been written.')

print("")
print("---------------------------------------------------------")
print("The list of categories are (id, name):")
for cat in db.dataset['categories']:
    label = cat['id'] + 1
    if label <= 27:
        print(f'({label}, {cat["name"]})')