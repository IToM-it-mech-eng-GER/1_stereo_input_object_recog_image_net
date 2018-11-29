import pickle
import pandas as pd

import numpy as np
import os, shutil
import scipy.misc
import logging
import argparse
import itertools

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#from keras import backend as K


# Quelle: Deep learning with python von FRANÇOIS CHOLLET Seite:137    
def plot_history(history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    
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

# Quelle: Deep learning with python von FRANÇOIS CHOLLET Seite:137    
def history_to_csv(history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    

    data = []
    for i in range(len(acc)):
        data.append({'.no:':i, \
                     'acc': acc[i], \
                     'val_acc': val_acc[i], \
                     'loss': loss[i], \
                     'val_loss': val_loss[i]})

    results_df = pd.DataFrame(data)
    results_df.to_csv('../results/trainHistoryDict_14_01.csv', index=False)

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

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, normalized=False): # https://gist.github.com/zachguo/10296432
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
            if (normalized==True):
                cell = "%{0}.2f".format(columnwidth) % cm[i, j]
            else:
                cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):
# https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=90)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')

if __name__ == "__main__" :
	
    CONFIG = 7 # default
    EPOCHS = 40 # default

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--epochs", default=EPOCHS)
 
    args = parser.parse_args()

    config=int(args.config)
    epochs = int(args.epochs)

    predfile = 'predictionDict_{:02d}_e{:02d}.hst'.format(config, epochs)

    with open(predfile, 'rb') as predictions_f:
        predictions = pickle.load(predictions_f)# load file content as mydict
    
    #acc = K.mean(K.equal(predictions['y_true'], predictions['y_pred']))
    #acc = K.mean(K.equal(K.argmax(Y_true, axis=-1), K.argmax(Y_pred, axis=-1)))
    #print('Accuracy ')
    #print(acc)
 
    print('Class indices')
    print(predictions['class_indices'])
    
    print('Classification Report')
    #class_names = ['Cats', 'Dogs', 'Horse']
    #print(classification_report(y_true, y_pred, target_names=target_names))
    cl_rep  = classification_report(predictions['y_true'], predictions['y_pred'], target_names=predictions['class_names'])
    print(cl_rep)

    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = 6
    fig_size[1] = 12
    plt.rcParams.update({'figure.figsize' : fig_size})

    plot_classification_report(cl_rep)
    fig = plt.gcf()
    pltfile = 'class_report_{:02d}_e{:02d}.png'.format(config, epochs)
    fig.savefig(pltfile)


    # Compute confusion matrix
    #cnf_matrix = confusion_matrix(classes_pred, y_pred) #, labels=target_names)
    cnf_matrix = confusion_matrix(predictions['y_true'], predictions['y_pred']) #, labels=target_names)
    np.set_printoptions(precision=0) # not normalized figures
    print('print_cm not normalized')
    print_cm(cnf_matrix, predictions['class_names'], normalized=False)
    
    
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2) # print normalized figures
    print('print_cm normalized')
    print_cm(cnf_matrix_norm, predictions['class_names'], normalized=True)

    # Set figure width to 1400 and height to 1200
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = 14
    fig_size[1] = 12
    plt.rcParams.update({'figure.figsize' : fig_size})
 
    '''
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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(True)

    # save omitted, because not scalable
    fig = plt.gcf()
    fig.savefig('cnf_matrix_color.png')
    

    #print(cnf_matrix, file=open("conf_matrix", "a"))
    #print(cnf_matrix)
    '''


    # Plot non-normalized confusion matrix
    plt.figure()
    #plt.tick_params(labelsize=8)
    
    plt.rcParams.update({'font.size': 6})
    plot_confusion_matrix(cnf_matrix, classes=predictions['class_names'],
                        title='Confusion matrix, without normalization')
    # save omitted, because not scalable
    fig = plt.gcf()
    pltfile = 'cnf_matrix_wout_norm_{:02d}_e{:02d}.png'.format(config, epochs)
    fig.savefig(pltfile)

    # Plot normalized confusion matrix
    plt.figure()
    #plt.tick_params(labelsize=8)
    plt.rcParams.update({'font.size': 6})

    plot_confusion_matrix(cnf_matrix, classes=predictions['class_names'], normalize=True,
                        title='Normalized confusion matrix')
    # save omitted, because not scalable
    fig = plt.gcf()
    pltfile = 'cnf_matrix_with_norm_{:02d}_e{:02d}.png'.format(config, epochs)
    fig.savefig(pltfile)
    
    plt.show()
