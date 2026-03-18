import sys
import numpy as np
import cv2
from pycocotools.coco import COCO

file = sys.argv[1]
dest = sys.argv[2]
db = COCO(file)
im_ids = db.getImgIds()

for im_id in im_ids:
	#print(db.imgs[im_id]['file_name'])
	ann_ids = np.array(db.getAnnIds(imgIds=im_id))
	
	cat_ids = []
	for i in ann_ids:
		cat_ids.append(db.anns[i]['category_id']+1)
	ann_ids = ann_ids[np.argsort(cat_ids)]
	
	mask = None
	for i in ann_ids:
		m = (db.anns[i]['category_id']+1)*db.annToMask(db.anns[i])
		mm = (m>0).astype(np.uint8)
		if mask is None:
			mask = m
		else:
			mask = mask*(1-mm) + m
			
	name = db.imgs[im_id]['file_name'].split('.')[0]+'_seg.png'
	cv2.imwrite(dest+'/'+name, mask)
	print('file ' + name + ' has been written.')
	
	#cv2.imshow("mask", (mask*5.3).astype(np.uint8))
	#cv2.waitKey(0)

print("")
print("---------------------------------------------------------")
print("The list of categories are (id, name):")
for cat in db.dataset['categories']:
	print('('+str(cat['id']+1)+', '+cat['name']+')')