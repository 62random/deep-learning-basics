import matplotlib.pyplot as plt
from keras.datasets import mnist 
from keras.datasets import cifar10
from math import ceil
from keras.utils import plot_model
from PIL.Image import fromarray
import seaborn as sns
import pandas as pd


# Visualize a specified set of mnist records
def visualize_mnist(indices):
    (x_train, _), _ = mnist.load_data()
    cols = 5
    rows = ceil(len(indices)/float(cols))
    for i in range(len(indices)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x_train[indices[i]], cmap=plt.get_cmap('gray'))


# Visualize a specified set of cifar10 records
def visualize_cifar10(indices):
    (x_train, _), _ = cifar10.load_data()
    cols = 4
    rows = ceil(len(indices)/float(cols))
    for i in range(len(indices)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fromarray(x_train[indices[i]]))


# Wrapper for the Keras model visualization utility
def print_model(model, file):
    plot_model(model,
               to_file=file,
               show_shapes=True,
               show_layer_names=True
               )


# Visualize evolution of a specified set of metrics
# during a model's training
def plot_history_metrics(history, metrics):
    for i in range(len(metrics)):
        plt.plot(history.history[metrics[i]])
        plt.plot(history.history[f'val_{metrics[i]}'])
        plt.title(metrics[i])
        plt.ylabel(metrics[i])
        plt.xlabel('epoch')
        plt.legend([metrics[i], f'val_{metrics[i]}'], loc='upper left')
        plt.show()


def analyse_accuracy(histories, names):
    accs = [max(x.history['accuracy']) for x in histories]
    val_accs = [max(x.history['val_accuracy']) for x in histories]

    df = pd.DataFrame({'version': names * 2,
                       'accuracy': accs + val_accs,
                       'when': ['train'] * len(names) + ['validation'] * len(names)
                       })

    sns.barplot(x='version', y='accuracy', hue='when', data=df)
    plt.ylim(df['accuracy'].min() - 0.05, 1)
    plt.legend(loc='lower left')
