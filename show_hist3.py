import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import argparse
import os


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
def history_to_csv(history, hist):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    

    data = []
    for i in range(len(acc)):
        data.append({'#':i+1, \
                     'acc': acc[i], \
                     'val_acc': val_acc[i], \
                     'loss': loss[i], \
                     'val_loss': val_loss[i]})

    results_df = pd.DataFrame(data)
    results_df.to_csv(hist + '.csv', index=False)


if __name__ == "__main__" :
	#history.histoy['loss'][5]
    # reading history.history:
    # Filenamen-Muster trainHistoryDict_13_e20_c05.hst
    # params for csv output:    -csvonly --hist trainHistoryDict_01_e20_c01.hst
    # params for plot and csv:  --hist trainHistoryDict_01_e20_c01.hst

    parser = argparse.ArgumentParser()
    parser.add_argument('--hist')
    parser.add_argument('-csvonly', action='store_true')
 
    args = parser.parse_args()

    hist=args.hist
    csvonly=args.csvonly

    if not (os.path.isfile(hist)):
        print('No hist file ' + hist + ' found')
        quit()
    history_f = open(hist, 'rb') # rb==read binary
    history_history = pickle.load(history_f)# load file content as mydict
    if not (csvonly):
        pprint(history_history)
        plot_history(history_history)
    history_to_csv(history_history, hist)
    history_f.close()
    # history_history['loss'][99] will return a loss of the model in a 100th epochof training