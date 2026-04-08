from pathlib import Path

ROOT = Path("/Users/Rafael/Downloads")
TRAIN_IMG= ROOT/"val_test2020/test/"
TEST_IMG =ROOT/"val_test2020/test/"
TRAIN_MASK = ROOT/"mask_val/"
TEST_MASK= ROOT/"mask_val/"
MODELS = ROOT/"models/"
RESULTS = ROOT/"results/"
ANNOTATIONS_TRAIN = "/Users/Rafael/Downloads/instances_attributes_val2020.json"
ANNOTATIONS_TEST = "/Users/Rafael/Downloads/instances_attributes_val2020.json"
FABRICS = ROOT/"textures/"