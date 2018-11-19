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
