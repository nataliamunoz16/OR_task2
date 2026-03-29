Instructions to convert Fashionpedia json file to png semantic segments
=======================================================================

Fashionpedia keeps all ground truth data in the json file. The segmentation data is in polygons and we want to convert them into segmentation images which are more convinient to work with. Segmentation images will be saved in png format in which each pixel takes its semantic label (0 means background). The shape of the segment image is (Height, Width, 1).

To convert the data, you must download and install COCOAPI.
- You can download or clone the cocoapi from: https://github.com/cocodataset/cocoapi
- Go to PythonAPI and run "setup.py" as instructed in the file.
NOTE: if you encounter troubles installing cocoapi on windows, check here: https://github.com/cocodataset/cocoapi/issues/51

Then move "fashionpedia.py", which is in the same place as this readme, to the PythonAPI directory inside cocoapi. Finally, run it as following:

python fashionpedia.py path/to/jason_file path/to/the_directory_you_want_to_save_segmentations

NOTE: there are two json files, one for train and the other for validation. Please use the validation data to evaluate and report your metrics.