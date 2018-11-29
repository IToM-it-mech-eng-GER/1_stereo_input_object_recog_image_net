import numpy as np
import cv2 as cv2
import os, shutil
import scipy.misc
import logging
import argparse
import itertools

from smallnorb.norb_srgb import SmallNORBDataset
from washington_rgbd.washington_rgbd import WashingtonRGBD
from generator.generator import Generator

import pandas as pd
import pickle


from keras import backend as K
from keras.utils import to_categorical
#FA from keras.layers import Convolution2D, Cropping2D, MaxPooling2D, Concatenate
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D, Concatenate, ZeroPadding2D# , Merge
from keras.models import Sequential
from keras.models import Model
#FA from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam
from keras.applications import VGG16
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
#from tensorflow.contrib.learn.python.learn import trainable
#np.random.seed(42)

load_VGG16_weights = True
vgg16_weightsfile = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
RGBD = 'rgbd'
NORB = 'norb'
RGBDW = 'rgbdw' # whole rgbd image in single net
SEED = 42


#switch between single and double headed model        
def build_cnn(dataset, doublehead, cropping_l, cropping_r, classes, learning_rate):
    if (doublehead):
        # benutze double head Net
        return build_double_cnn(dataset, cropping_l, cropping_r, classes, learning_rate)
    else:
        # benutze single head Net
        return build_single_cnn(dataset, cropping_l, classes, learning_rate)

        
def build_double_cnn(dataset, cropping_l, cropping_r, classes, learning_rate):
    strides_val = (2, 2)
    pool_size_val=(2, 2)
    if (dataset == RGBD):
        shape = (224, 448, 3) # target size des ImageDataGenerators
        left_input = Input(shape=shape, name="left_input")
        # cropping
        x = Cropping2D(cropping=cropping_l, input_shape=shape, trainable=False)                                         (left_input)
        x = Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False, name='block1_conv1')          (x)
    
    else:
        shape = (224, 224, 3) # target size des ImageDataGenerators
        left_input = Input(shape=shape, name="left_input")
        x = Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False, name='block1_conv1')          (left_input)
    
    # FAPI left model
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False, name='block1_conv2')          (x)
    x = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block1_pool')                       (x)

    x = Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False, name='block2_conv1')         (x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False, name='block2_conv2')         (x)
    x = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block2_pool')                       (x)

    x = Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False, name='block3_conv1')         (x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False, name='block3_conv2')         (x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False, name='block3_conv3')         (x)
    x = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block3_pool')                       (x)

    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block4_conv1')         (x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block4_conv2')         (x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block4_conv3')         (x)
    x = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block4_pool')                       (x)

    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block5_conv1')         (x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block5_conv2')         (x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block5_conv3')         (x)
    x = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block5_pool')                       (x)

    x = Flatten(name='flatten')                                                                                     (x)
    x = Dense(4096, activation='relu', trainable=False, name='fc1')                                                 (x)
    x = Dense(4096, activation='relu', trainable=False, name='fc2')                                                 (x)
    # Using original class count for loading original weights 
    x = Dense(1000, activation='softmax', trainable=True, name='vgg_predictions')                                   (x)

    left_model = Model(inputs=left_input, outputs=x)
    # Loading original model weights
    if (load_VGG16_weights == True) :
        left_model.load_weights(vgg16_weightsfile, by_name=True)

    for layer in left_model.layers:
        layer.name = str('l_') + layer.name

    # FAPI right model
    if (dataset == RGBD):
        shape = (224, 448, 3) # target size des ImageDataGenerators
        right_input = Input(shape=shape, name="right_input")
        # cropping
        y = Cropping2D(cropping=cropping_r, input_shape=shape, trainable=False)                                         (right_input)
        y = Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False, name='block1_conv1')          (y)
    
    else:
        shape = (224, 224, 3) # target size des ImageDataGenerators
        right_input = Input(shape=shape, name="right_input")
        y = Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False, name='block1_conv1')          (right_input)
 
    
    y = Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False, name='block1_conv2')          (y)
    y = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block1_pool')                       (y)

    y = Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False, name='block2_conv1')         (y)
    y = Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False, name='block2_conv2')         (y)
    y = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block2_pool')                       (y)

    y = Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False, name='block3_conv1')         (y)
    y = Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False, name='block3_conv2')         (y)
    y = Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False, name='block3_conv3')         (y)
    y = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block3_pool')                       (y)

    y = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block4_conv1')         (y)
    y = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block4_conv2')         (y)
    y = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block4_conv3')         (y)
    y = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block4_pool')                       (y)

    y = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block5_conv1')         (y)
    y = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block5_conv2')         (y)
    y = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block5_conv3')         (y)
    y = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block5_pool')                       (y)

    y = Flatten(name='flatten')                                                                                     (y)
    y = Dense(4096, activation='relu', trainable=False, name='fc1')                                                 (y)
    y = Dense(4096, activation='relu', trainable=False, name='fc2')                                                 (y)
    # Using original class count for loading original weights 
    y = Dense(1000, activation='softmax', trainable=True, name='vgg_predictions')                                   (y)

    right_model = Model(inputs=right_input, outputs=y)

    # Loading second modelweights
    if (load_VGG16_weights == True) :
        right_model.load_weights(vgg16_weightsfile, by_name=True)
    
    for layer in right_model.layers:
        layer.name = str('r_') + layer.name

    # FAPI
    # define a new output layer to connect with the last fc layer in vgg
    # thanks to joelthchao https://github.com/fchollet/keras/issues/2371
    x = left_model.layers[-2].output # drop original classification layer left
    y = right_model.layers[-2].output # drop original classification layer right
    merge = concatenate([x, y])
    
    output_layer = Dense(4096, activation='relu', trainable=True, name='dense_merged')(merge) 
    output_layer = Dense(classes, activation='softmax', trainable=True, name='predictions')(output_layer) 
 
    model_tail = Model(inputs=[left_input, right_input], outputs=output_layer)

    model_tail.summary()

    #V06 model_tail.compile(optimizer=Adam(lr=0.001), # V06
    model_tail.compile(optimizer=Adam(lr=learning_rate), # V07-31 Default 0.0001
                        loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model_tail

# single input model
def build_single_cnn(dataset, cropping_l, classes, learning_rate):
    strides_val = (2, 2)
    pool_size_val=(2, 2)
    if (dataset == RGBD):
        shape = (224, 448, 3) # target size des ImageDataGenerators
        left_input = Input(shape=shape, name="left_input")
        # cropping
        x = Cropping2D(cropping=cropping_l, input_shape=shape, trainable=False)                                         (left_input)
        x = Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False, name='block1_conv1')          (x)
    elif (dataset == RGBDW):
        shape = (224, 224, 3) # target size des ImageDataGenerators, keras will resize images with shape (224, 448)
        left_input = Input(shape=shape, name="left_input")
        x = Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False, name='block1_conv1')          (left_input)   
    
    else:
        shape = (224, 224, 3) # target size des ImageDataGenerators
        left_input = Input(shape=shape, name="left_input")
        x = Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False, name='block1_conv1')          (left_input)   
    
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False, name='block1_conv2')          (x)
    x = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block1_pool')                       (x)

    x = Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False, name='block2_conv1')         (x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False, name='block2_conv2')         (x)
    x = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block2_pool')                       (x)

    x = Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False, name='block3_conv1')         (x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False, name='block3_conv2')         (x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False, name='block3_conv3')         (x)
    x = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block3_pool')                       (x)

    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block4_conv1')         (x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block4_conv2')         (x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block4_conv3')         (x)
    x = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block4_pool')                       (x)

    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block5_conv1')         (x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block5_conv2')         (x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False, name='block5_conv3')         (x)
    x = MaxPooling2D(pool_size_val, strides=strides_val, trainable=False, name='block5_pool')                       (x)

    x = Flatten(name='flatten')                                                                                     (x)
    x = Dense(4096, activation='relu', trainable=False, name='fc1')                                                 (x)
    x = Dense(4096, activation='relu', trainable=False, name='fc2')                                                 (x)
    # Using original class count for loading original weights 
    x = Dense(1000, activation='softmax', trainable=True, name='vgg_predictions')                                   (x)

    left_model = Model(inputs=left_input, outputs=x)
    
    # Loading original model weights
    if (load_VGG16_weights == True) :
        left_model.load_weights(vgg16_weightsfile, by_name=True)

    # define a new output layer to connect with the last fc layer in vgg
    # thanks to joelthchao https://github.com/fchollet/keras/issues/2371
    x = left_model.layers[-2].output
    output_layer = Dense(classes, activation='softmax', name='predictions')(x)

    # combine our model with the new output layer
    left_model = Model(inputs=left_model.input, outputs=output_layer)

    # Keras VGG16 SAPI Code https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py (deprecated)
    # Keras actual VGG16 Code with weights https://github.com/fchollet/deep-learning-models/releases
    # Original VGG16 written in Keras SAPI https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3 - NOT USED

    # retrieve the full Keras VGG model including imagenet weights (http://forums.fast.ai/t/difference-b-w-jeremys-vgg16-and-keras-built-in-vgg-16/881/15 - Richard Horton)
    
    left_model.summary()
    # compile the new model
    #left_model.compile(optimizer=Adam(lr=0.001),
    left_model.compile(optimizer=Adam(lr=learning_rate), # # V07-31 Default 0.0001
                        loss='categorical_crossentropy', metrics=['accuracy'])
    return left_model

def train_cnn(model, train_generator, validation_generator, steps_per_epoch=100, epochs_val=5, config=1, reload=0):
    # Learn
    history = model.fit_generator(  train_generator,
                                    verbose=1,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs_val,
                                    validation_data=validation_generator,
                                    validation_steps=steps_per_epoch/2, 
                                    shuffle=True) # for reproducible results
    
    # save new weights 
    if (reload > 0):
        epochs = epochs_val + reload
    else:
        epochs = epochs_val
    modelname = 'weights_{:02d}_{:02d}'.format(config, epochs)   
    model.save(modelname + '.h5')
    with open(modelname  + '.json', 'w') as text_file:
        print(model.to_json(), file=text_file)
    
    return history
    
# Quelle: Deep learning with python von FRANÇOIS CHOLLET Seite:137    

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'   #TM .2f or .1f
    thresh = cm.max() / 4. #TM 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None): # https://gist.github.com/zachguo/10296432
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
'''
    columnwidth = max([len(x) for x in labels])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t\\p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("  " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("  %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
'''

if __name__ == "__main__" :         
    # The good stuff! Load, preprocess, split, train, save!
    # Initialize the dataset from the folder in which
    # dataset archives have been uncompressed (ROOT_DEFAULT)

    ''' 
    # major use cases as samples
    
    1. train from scratch, use Activated==True configurations in config_net.csv (only possible for 1 config at a time, e.g. 4), train default 20 epochs, cycle no 1 default
    program 
    # creates weights_04_20.h5 and train_HistoryDict_04_e20_c01.hst
    
    2. train from scratch, do not use Activated==True, but use config #10, train 30 epochs, cycle no 1 default
    program --config 10 --epochs 30  
    # creates weights_10_30.h5 and train_HistoryDict_10_e30_c01.hst
    
    3. reload weights for config 18, from 20th epoch (for default 20 additional epochs -> in sum 40), previously trained (weights_18_20.h5 is necessary), cycle no 1 default
    program --reload 20 --config 18
    # creates weights_18_40.h5 and train_HistoryDict_18_e40_c01.hst
    
    4. train a certain amount of cycles (e.g. 5) with a dedicated configuration no (e.g. 7) (use bash loop)
    for i in 1 2 3 4 5
    do
    program --config 7 --cycle $i
    done
    # creates weights_07_20.h5 (no additional knowledge gain) and train_HistoryDict_07_e20_c01.hst to train_HistoryDict_07_e20_c05.hst (creating one hist-File for every cycle)

    5. predict on base of validation images for configuration 1 and weights 20 (use weights/weights_01_20.h5 file)
    program --config 1 --predict 20

    # Create a session with the above options specified.
    # Hint from https://michaelblogscode.wordpress.com/2017/10/10/reducing-and-profiling-gpu-memory-usage-in-keras-with-tensorflow-backend/
    k.tensorflow_backend.set_session(tf.Session(config=config))
    '''
    # according to https://github.com/keras-team/keras/issues/2102
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))
    
    # initialize random generators with a static seed
    # K.tf.set_random_seed(1234)
    np.random.seed(1234)

    CONFIG_FILE = 'config_net.csv'

    NORB_ROOT_DEFAULT = '../norb_dataset'
    NORB_CSV_DATASET = '../norb_dataset_csv'
    NORB_MASTER_DEFAULT = '../norb_master_dir'
    NORB_DEST_DEFAULT = '../norb_to_process'
    NORB_CSV_MASTER_DATASET = os.path.join(NORB_CSV_DATASET, 'norb-dataset-master.csv')
    NORB_CSV_PREPARED_SET = os.path.join(NORB_CSV_DATASET, 'norb-dataset-prepared.csv')
    NORB_CSV_PREPARED_DIRS = os.path.join(NORB_CSV_DATASET, 'norb-dataset-prepared_dirs.csv')
    NORB_CSV_PROCESS_DIRS = os.path.join(NORB_CSV_DATASET, 'norb-dataset-process_dirs.csv')
    
    RGBD_ROOT_DEFAULT = '../rgbd-dataset'
    RGBD_CSV_DATA = '../rgbd-dataset_csv'
    RGBD_MASTER_DEFAULT = '../master_dir'
    RGBD_DEST_DEFAULT = '../data_to_process'
    RGBD_CSV_DATASET = os.path.join(RGBD_CSV_DATA, 'rgbd-dataset.csv')
    RGBD_CSV_AGGREGATED_DATASET = os.path.join(RGBD_CSV_DATA, 'rgbd-dataset-aggregated.csv')
    RGBD_CSV_MASTER_DATASET = os.path.join(RGBD_CSV_DATA, 'rgbd-dataset-master.csv')
    RGBD_CSV_TT_SPLIT_01 = os.path.join(RGBD_CSV_DATA, 'rgbd-dataset-ttsplit01.csv')
    RGBD_CSV_PREPARED_SET = os.path.join(RGBD_CSV_DATA, 'rgbd-dataset-prepared.csv')
    RGBD_CSV_PREPARED_DIRS = os.path.join(RGBD_CSV_DATA, 'rgbd-dataset-prepared_dirs.csv')
    RGBD_CSV_PROCESS_DIRS = os.path.join(RGBD_CSV_DATA, 'rgbd-dataset-process_dirs.csv')
    
    DEST_SIZE = 224
    SPLIT_RATE = 0.2
    CONFIG = '0' # default: use config_net file
    CYCLE = 1 # default: first cycle
    EPOCHS_VAL = 20 # mit VGG und Adam 20, vorher 50 (10 waren zu wenig, 100 zu viel)
    BATCH_SIZE = 20
    RELOAD = 0 # kein Reload Model mit weights von vorherigen Epochen machen
    LR = 0.0001 # default learning rate
    PREDICT = 0 # no prediction
    PREDICTION_SLICE = 0.99 # slice of validation files to be used for predictions

    logging.basicConfig(level=logging.INFO)
    # print(os.getcwd()) # debug

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--cycle", default=CYCLE)
    parser.add_argument("--epochs", default=EPOCHS_VAL)
    parser.add_argument("--batch_size", default=BATCH_SIZE)
    parser.add_argument("--reload", default=RELOAD)
    parser.add_argument("--lr", default=LR)
    parser.add_argument("--predict", default=PREDICT)

    args = parser.parse_args()

    #config_main=7 #10 #7 #1 #DEBUG args.config
    config_main=args.config
    cycle = int(args.cycle)
    epochs_val = int(args.epochs)
    batch_size = int(args.batch_size)
    reload = int(args.reload)
    learning_rate = float(args.lr)
    #predict = 40 #DEBUG int(args.predict)
    predict = int(args.predict)
    if (predict > 0):
        reload = predict

    if (config_main == '0'):
        check_activated = True # check only config_net file 
    else:
        check_activated = False

    cropping_2D = ((0, 0), (0, 224))
    cropping_3D = ((0, 0), (224, 0))

    netconfig_df = pd.read_csv(CONFIG_FILE) 
    for i in range(len(netconfig_df.index)):
        current_config_row = netconfig_df.iloc[[i]]
        activated = current_config_row.Activated.values[0]
        config = current_config_row.Config.values[0]
        
        if (check_activated == True):
            if (activated == 'n'):
                continue
        else: 
            if (str(config_main) != str(config)):
                continue
        # activated or config_main

        current_dataset = current_config_row.Dataset.values[0]
        if (current_dataset == NORB):
            # evaluate process dirs file 
            the_dataset = SmallNORBDataset(dataset_root=NORB_ROOT_DEFAULT,
                                        csv_dataset=NORB_CSV_DATASET,
                                        csv_master_dataset=NORB_CSV_MASTER_DATASET,
                                        csv_prepared_set=NORB_CSV_PREPARED_SET,
                                        csv_prepared_dirs=NORB_CSV_PREPARED_DIRS,
                                        csv_process_dirs=NORB_CSV_PROCESS_DIRS,
                                        master_default=NORB_MASTER_DEFAULT,
                                        dest_default=NORB_DEST_DEFAULT,
                                        dest_size=DEST_SIZE)

            the_dataset.logger.info('>-----------------------------------------------')

            dir_df = the_dataset.get_process_dirs()#.astype(str) # TM1
        else:
            # Washington (also for whole image)
            the_dataset = WashingtonRGBD(root_dir=RGBD_ROOT_DEFAULT,
                                        csv_dataset=RGBD_CSV_DATASET,
                                        csv_aggregated_dataset=RGBD_CSV_AGGREGATED_DATASET,
                                        csv_master_dataset=RGBD_CSV_MASTER_DATASET,
                                        csv_tt_split_01=RGBD_CSV_TT_SPLIT_01,
                                        csv_prepared_set=RGBD_CSV_PREPARED_SET,
                                        csv_prepared_dirs=RGBD_CSV_PREPARED_DIRS,
                                        csv_process_dirs=RGBD_CSV_PROCESS_DIRS,
                                        master_default=RGBD_MASTER_DEFAULT,
                                        dest_default=RGBD_DEST_DEFAULT,
                                        dest_size=DEST_SIZE,
                                        split_rate=SPLIT_RATE)

            print('>-----------------------------------------------')
            dir_df = the_dataset.get_process_dirs()#.astype(str) # TM1
        #config = current_config_row.Config.values[0]
        l_head_input = current_config_row.L_head_input.values[0]
        r_head_input = current_config_row.R_head_input.values[0]
        doublehead_str = current_config_row.Doublehead.values[0]
        filled_set_val = current_config_row.Filled_set.values[0]
        source_left = current_config_row.Source_left.values[0]#.astype(str) # TM1
        source_right = current_config_row.Source_right.values[0]#.astype(str) # TM1
        description = current_config_row.Description.values[0]
        the_dataset.logger.info('-----------------------------------------------')
        the_dataset.logger.info('Config #' + str(config) + ' - ' + description)

        if (l_head_input == '2D'): 
            cropping_l_head = cropping_2D
        elif (l_head_input == '3D'):
            cropping_l_head = cropping_3D
        else: 
            cropping_l_head = None # NORB or RGDBW
            
        if (r_head_input == '2D' or r_head_input == 'n'): 
            cropping_r_head = cropping_2D
        elif (r_head_input == '3D'):
            cropping_r_head = cropping_3D
        else: 
            cropping_r_head = None
            
        if (doublehead_str == 'n'): 
            doublehead=False
        else:
            doublehead=True
        
        if (filled_set_val == 'y'):
            filled_set = 'filled'
        elif (filled_set_val == 'n'):
            filled_set = 'unfilled'
        else:
            filled_set = None
        
        # Train und Valid Dir heraussuchen entsprechend degree und filled_set:
        # source_left enthält degree für linken Head, source_right degree für rechten Head
        if (current_dataset == NORB):
            process_dir_row_l = dir_df[ (dir_df.degree == int(source_left))]
            #TM1process_dir_row_l = dir_df[ (dir_df.degree == str(source_left))]
        else: # Washington (also for whole image)
            process_dir_row_l = dir_df[ (dir_df.degree == int(source_left))
                                & (dir_df.filled_set == filled_set)]
            #TM1process_dir_row_l = dir_df[ (dir_df.degree == int(source_left))
            #TM1                    & (dir_df.filled_set == filled_set)]
        train_dir_left = process_dir_row_l.train_dir.values[0]
        valid_dir_left = process_dir_row_l.valid_dir.values[0]
        
        if (doublehead):
            if (current_dataset == NORB):
                process_dir_row_r = dir_df[(dir_df.degree == int(source_right))]
                #TM1process_dir_row_r = dir_df[(dir_df.degree == str(source_right))]
            else:
                process_dir_row_r = dir_df[(dir_df.degree == int(source_right))
                                & (dir_df.filled_set == filled_set)]
                #TM1process_dir_row_r = dir_df[(dir_df.degree == str(float(source_right)))
                #TM1            & (dir_df.filled_set == filled_set)]
            train_dir_right = process_dir_row_r.train_dir.values[0]
            valid_dir_right = process_dir_row_r.valid_dir.values[0]
        else:
            train_dir_right = ''
            valid_dir_right = ''
        the_dataset.logger.info('train_dir_left: ' + train_dir_left)
        the_dataset.logger.info('train_dir_right: ' + train_dir_right)
        the_dataset.logger.info('valid_dir_left: ' + valid_dir_left)
        the_dataset.logger.info('valid_dir_right: ' + valid_dir_right)


 
        # get metadata about dataset
        image_count, class_count = the_dataset.get_train_image_class_count()
        the_dataset.logger.info('-> Image count for training: ' + str(image_count))
        the_dataset.logger.info('-> Class count for training: ' + str(class_count))

        v_image_count, v_class_count = the_dataset.get_valid_image_class_count()
        the_dataset.logger.info('=> Image count for validation: ' + str(v_image_count))
        the_dataset.logger.info('=> Class count for validation: ' + str(v_class_count))


        #train_generator, validation_generator = load_data_generators(train_dir, validation_dir)
        generator = Generator(seed=SEED)
        # input_train_generator, input_validation_generator = 
        generator.load_data_generators(current_dataset, doublehead, train_dir_left, valid_dir_left, \
                                                        train_dir_right, valid_dir_right, batch_size)

        if (reload == 0):
            # build the net
            model = build_cnn(current_dataset, doublehead=doublehead, cropping_l=cropping_l_head, cropping_r=cropping_r_head, classes=class_count, learning_rate=learning_rate)
        else:
            # reload the net
            modelfile = 'weights_{:02d}_{:02d}.h5'.format(config, reload)   
            modelname = os.path.join('weights', modelfile)
            if not os.path.isfile(modelname):
                the_dataset.logger.error('No such file: ' + modelname + ' -> quit program')
                quit()
            else:
                the_dataset.logger.info('Loading model: ' + modelname)
            model = load_model(modelname) # gebildet aus Übergabeparametern (args)
            model.summary()

        if (predict == 0):
            # training
            steps_per_epoch = int(image_count / batch_size)
            the_dataset.logger.info('steps per epoch: ' + str(steps_per_epoch))
            history = train_cnn(model, generator.train_generator, generator.valid_generator, steps_per_epoch, epochs_val, config, reload)
            
            # save history
            epochs = epochs_val + reload
            histfile = 'trainHistoryDict_{:02d}_e{:02d}_c{:02d}.hst'.format(config, epochs, cycle)
            #picklefile = '{:06d}_{}_{:02d}_rt.jpg'.format(i, category, instance)
            with open(histfile, 'wb') as file_pi:
                pickle.dump(history.history, file_pi) # history.history['loss'][99]
        else:
            # Predict and print confusion matrix
            pred_count = int(v_image_count*PREDICTION_SLICE)
            #24160nok #24180nok #24120ok #24100ok #24200nok #24000ok #24220nok #300*batch_size #DEBUG
            print('Predicting approx. ' + str(pred_count) + ' images')
            print('Loops ' + str(pred_count // batch_size + 1))
            Y_pred = model.predict_generator(generator.valid_generator, pred_count // batch_size + 1)
            print('Y_pred.shape ')
            print(Y_pred.shape)
            y_pred = np.argmax(Y_pred, axis=1)
            
            # static information about images (before prediction)
            classes= generator.get_classes() # true classes list of the valid set not shuffled
            print('Länge Liste classes ' + str(len(classes)))
            filenames = generator.get_filenames() # true filenames list of the valid set not shuffled
            print('Länge Liste filenames ' + str(len(filenames)))
            index_array = generator.get_index_array() # shuffled index list
            print('Länge Liste index_array ' + str(len(index_array)))
            class_indices = generator.get_class_indices() # dict of key (Index) and classname for valid set

            length = len(y_pred) # length of prediction list (count of images)
            print('Länge Liste y_pred ' + str(len(y_pred)))
            
            classes_true = []
            for i in range(length):
                classes_true.append(classes[index_array[i]]) # get true class for the first n (lenght) images of valid set
            
            # get true class list as np array
            y_true = np.asarray(classes_true) 
            #Y_true = to_categorical(y_true, v_class_count) # not necessary
            #print(Y_true)

            filenames_pred = []
            for i in range(length):
                filenames_pred.append(filenames[index_array[i]])
            
            # get class names list
            class_names = []
            for key, id in class_indices.items():
                #class_names.append(key[:9])
                class_names.append(key)


            predictions = {}
            predictions['y_true'] = y_true
            predictions['y_pred'] = y_pred
            predictions['class_names'] = class_names
            predictions['class_indices'] = class_indices

            predfile = 'predictionDict_{:02d}_e{:02d}.hst'.format(config, predict)
            with open(predfile, 'wb') as file_pi:
                pickle.dump(predictions, file_pi) 


            print('Predictions finished')

            print('Classification Report')
            #class_names = ['Cats', 'Dogs', 'Horse']
            #print(classification_report(y_true, y_pred, target_names=target_names))
            cl_rep  = classification_report(predictions['y_true'], predictions['y_pred'], target_names=predictions['class_names'])
            print(cl_rep)
            '''
            # Compute confusion matrix
            #cnf_matrix = confusion_matrix(classes_pred, y_pred) #, labels=target_names)
            cnf_matrix = confusion_matrix(predictions['y_true'], predictions['y_pred']) #, labels=target_names)
            np.set_printoptions(precision=0) # not normalized figures
            print('print_cm not normalized')
            print_cm(cnf_matrix, predictions['class_names'])
            
            
            cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(precision=2) # print normalized figures
            print('print_cm normalized')
            print_cm(cnf_matrix_norm, predictions['class_names'])

            
            #confmat=np.random.rand(90,90)
            plt.figure()
            clen = len(predictions['class_names'])
            ticks=np.linspace(0, clen-1,num=clen)
            plt.rcParams.update({'font.size': 6})
            #plt.tick_params(labelsize=8)
            
            plt.imshow(cnf_matrix_norm, interpolation='none')
            plt.colorbar()
            plt.xticks(ticks,fontsize=6)
            plt.yticks(ticks,fontsize=6)
            plt.grid(True)

            # save
            fig = plt.gcf()
            fig.savefig('cnf_matrix_color.pdf')
            #plt.show()

            #print(cnf_matrix, file=open("conf_matrix", "a"))
            #print(cnf_matrix)



            # Plot non-normalized confusion matrix
            plt.figure()
            #plt.tick_params(labelsize=8)
            
            plt.rcParams.update({'font.size': 6})
            plot_confusion_matrix(cnf_matrix, classes=predictions['class_names'],
                                title='Confusion matrix, without normalization')
            # save
            fig = plt.gcf()
            fig.savefig('cnf_matrix_wout_norm.pdf')

            # Plot normalized confusion matrix
            plt.figure()
            #plt.tick_params(labelsize=8)
            plt.rcParams.update({'font.size': 6})

            plot_confusion_matrix(cnf_matrix, classes=predictions['class_names'], normalize=True,
                                title='Normalized confusion matrix')
            # save
            fig = plt.gcf()
            fig.savefig('cnf_matrix_with_norm.pdf')
            plt.show()
            '''
            '''
            print('Classification Report')
            #class_names = ['Cats', 'Dogs', 'Horse']
            #print(classification_report(y_true, y_pred, target_names=target_names))
            cl_rep  = classification_report(y_true, y_pred, target_names=class_names)
            print(cl_rep)

            # Compute confusion matrix
            #cnf_matrix = confusion_matrix(classes_pred, y_pred) #, labels=target_names)
            cnf_matrix = confusion_matrix(y_true, y_pred) #, labels=target_names)
            np.set_printoptions(precision=0) # not normalized figures
            print('print_cm not normalized')
            print_cm(cnf_matrix, class_names)
            
            
            cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(precision=2) # print normalized figures
            print('print_cm normalized')
            print_cm(cnf_matrix_norm, class_names)

            
            #confmat=np.random.rand(90,90)
            plt.figure()
            ticks=np.linspace(0, 50,num=51)
            plt.rcParams.update({'font.size': 6})
            #plt.tick_params(labelsize=8)
            
            plt.imshow(cnf_matrix_norm, interpolation='none')
            plt.colorbar()
            plt.xticks(ticks,fontsize=6)
            plt.yticks(ticks,fontsize=6)
            plt.grid(True)

            # save
            fig = plt.gcf()
            fig.savefig('cnf_matrix_color.pdf')
            #plt.show()

            #print(cnf_matrix, file=open("conf_matrix", "a"))
            #print(cnf_matrix)



            # Plot non-normalized confusion matrix
            plt.figure()
            #plt.tick_params(labelsize=8)
            
            plt.rcParams.update({'font.size': 6})
            plot_confusion_matrix(cnf_matrix, classes=class_names,
                                title='Confusion matrix, without normalization')
            # save
            fig = plt.gcf()
            fig.savefig('cnf_matrix_wout_norm.pdf')

            # Plot normalized confusion matrix
            plt.figure()
            #plt.tick_params(labelsize=8)
            plt.rcParams.update({'font.size': 6})

            plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                title='Normalized confusion matrix')
            # save
            fig = plt.gcf()
            fig.savefig('cnf_matrix_with_norm.pdf')
            plt.show()

            '''

            # acc = K.mean(K.equal(y_true, y_pred))
            #acc = K.mean(K.equal(K.argmax(Y_true, axis=-1), K.argmax(Y_pred, axis=-1)))
            #print('Accuracy ')
            #print(acc)
        '''
        # reading history.history:
        history_f = open('trainHistoryDict', 'rb')   # rb==read binary
        history_history = pickle.load(history_f)         # load file content as mydict
        history_f.close() 
        # history_history['loss'][99] will return a loss of the model in a 100th epoch of training                      
        '''
        # Not for batch!
        #plot_history(history)

        # close session (and free memory) for next configuration
        K.clear_session()

        
        
