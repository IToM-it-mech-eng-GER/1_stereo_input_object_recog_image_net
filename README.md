# 1_stereo_input_object_recog_image_net
Simple code to use pretrained stereo_input_object_recognition_net

Setup:
writen with python 3.6, Tensorflow, Keras

General construction:
used standard VGG16 pretrained net with weights-file
cutting of last dense layer
doubling net
sticking two VGGs with third untrained dense-layer-structure

Datasets:
tested with Washington RGBG and NORB Dataset

Folder structure of inputdata:
one Folder with all Pictures in sorted-by-object-folders inside

weights-file from:
https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5

folderstructure:

mainfolder -> code -> recog_cnn.py
mainfolder -> code -> show_hist.py
mainfolder -> code -> smallnorb -> norb_srgb.py
mainfolder -> code -> washington_rgbd -> washington_rgbd.py
mainfolder -> results -> .hst 
mainfolder -> norb_dataset -> class1 -> object1_class1 -> .png
mainfolder -> rgb_dataset -> class1 -> object1_class1 -> .png
