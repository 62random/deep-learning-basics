import matplotlib.pyplot as plt
from keras.datasets import mnist 
from keras.datasets import cifar10
from keras.datasets import fashion_mnist
from math import ceil
from keras.utils import plot_model
from PIL.Image import fromarray
import seaborn as sns
import pandas as pd
import numpy as np


# Translate the fashion-mnist labels
def fashion_label(x):
    if x == 0:
        return 'T-shirt/top'
    elif x == 1:
        return 'Trouser'
    elif x == 2:
        return 'Pullover'
    elif x == 3:
        return 'Dress'
    elif x == 4:
        return 'Coat'
    elif x == 5:
        return 'Sandal'
    elif x == 6:
        return 'Shirt'
    elif x == 7:
        return 'Sneaker'
    elif x == 8:
        return 'Bag'
    elif x == 9:
        return 'Ankle boot'
    else:
        return 'wtf??'    

# Visualize a specified set of mnist records
def visualize_mnist(indices, data=None):
    if data is None:
        (x_train, _), _ = mnist.load_data()
    else:
        x_train = data
    
    cols = 5
    rows = ceil(len(indices)/float(cols))
    for i in range(len(indices)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x_train[indices[i]], cmap=plt.get_cmap('gray'))


# Visualize a specified set of mnist records
def visualize_fashion_mnist(indices):
    (x_train, _), _ = fashion_mnist.load_data()
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
    plt.figure()
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


# Function to visualize a model's prediction of an image
def visualize_prediction(model, x_test, index, layer_outputs_model, names, labels=None):
    plt.imshow(fromarray(((x_test[index])*255).astype(np.uint8)))
    plt.show()
    tensor = np.expand_dims(x_test[index], axis=0)
    if len(tensor.shape) == 3:
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2], 1)
    print('Prediction probabilities: ')
    print(model.predict(tensor))
    print('Predicted class: ')
    preds = model.predict_classes(tensor)
    if labels:
        print([labels[p] for p in preds])
    else:
        print(preds)

    print("Activations of Convolutional and Pooling layers:")
    activations = layer_outputs_model.predict(tensor)
    visualize_activations(activations, names)


# Function to visualize activations of a deep CNN
def visualize_activations(activations, names):
    images_per_line = 16

    for layer_name, layer_activation in zip(names, activations):
        n_features = layer_activation.shape[-1]
        width = layer_activation.shape[1]
        height = layer_activation.shape[2]
        n_lines = -(-n_features // images_per_line)

        display_grid = np.zeros((height * n_lines, images_per_line * width))
        for lin in range(n_lines):
            for col in range(images_per_line):
                if lin * images_per_line + col >= n_features:
                    break
                picture = layer_activation[0, :, :, lin * images_per_line + col]
                picture -= picture.mean()
                picture /= picture.std()
                picture *= 64
                picture += 128
                picture = np.clip(picture, 0, 255).astype('uint8')  # < 0 -> 0; > 255 -> 255
                display_grid[lin * height: (lin + 1) * height, col * width: (col + 1) * width] = picture
        h_scale = 1. / width
        v_scale = 1. / height
        plt.figure(figsize=(h_scale * display_grid.shape[1],
                            v_scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


# Function to visually compare input and predictions in autoencoders
# todo: implement display of generic number of images (not always 10)
def visualize_predictions(autoencoder, x_test):
    predicted_images = autoencoder.predict(x_test)
    
    # Original Images
    print("Original Images - first 10 images of test dataset")
    plt.figure(figsize=(30, 1))
    for i in range(10):
        ax = plt.subplot(1, 20, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
    # Display Decoded Images
    print("Decoded Images - predictions of first 10 images of test dataset")
    plt.figure(figsize=(30, 1))
    for i in range(10):
        ax = plt.subplot(1, 20, i+ 1)
        plt.imshow(predicted_images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    

    
##################################################################################
# VARIATIONAL AUTO ENCODERS
    
# Function to visualize  bidimensional latent vectors produced by a
# variational auto encoder
def plot_latent_v1(t_test, y_test):
    # grafico do latent vector t_test colorido pelos valores dos digitos nas imagens de input
    plt.figure(figsize=(8, 6))
    plt.scatter(t_test[:, 0], t_test[:, 1], marker='x', s=6.0, c=y_test,  cmap='brg')
    plt.colorbar();
    plt.show()

# Function to visualize  bidimensional latent vectors produced by a
# variational auto encoder - another version
def plot_latent_v2(t_test, y_test, fashion_mnist=False):   
    plt.figure(figsize=(8, 6))
    plt.scatter(t_test[:, 0], t_test[:, 1],s=1, c=y_test, cmap='brg')
    plt.colorbar();
    count=0;
    plt.tight_layout()
    for label, x, y in zip(y_test, t_test[:, 0], t_test[:, 1]):
        if count % 350 == 0:
            plt.annotate(fashion_label(int(label)) if fashion_mnist else str(int(label)),
                         xy=(x,y), 
                         color='black', 
                         weight='normal',
                         size=10,
                         bbox=dict(boxstyle="round4,pad=.5", fc="0.8"))
        count = count + 1
    plt.show()
    
    
##################################################################################