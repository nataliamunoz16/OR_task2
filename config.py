from pathlib import Path

ROOT = Path(r"/mnt/a27b1cbf-298a-4896-8528-84006d36fd9e/Nero/Documents/Acadèmic/UPC - MAMME/OR/P2/OR_task2")
TRAIN_IMG= ROOT/"coco/images/train/"
TEST_IMG =ROOT/"coco/images/val/"
TRAIN_MASK = ROOT/"coco/segmentations/seg_train/"
TEST_MASK= ROOT/"coco/segmentations/seg_val/"
MODELS = ROOT/"models/"
RESULTS = ROOT/"results/"