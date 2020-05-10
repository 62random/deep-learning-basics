import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD


# Wrapper function that calls other functions
# to create models defined in this module
def create_model(name, input_shape, num_classes, epochs=10, lr=1e-03):
    if name == 'mnist_mlp':
        return create_compile_mnist_mlp(np.prod(input_shape), num_classes)
    elif name == 'mnist_cnn_v1':
        return create_compile_mnist_cnn_v1(input_shape, num_classes, epochs, lr)
    elif name == 'mnist_cnn_v2':
        return create_compile_mnist_cnn_v2(input_shape, num_classes, epochs, lr)
    elif name == 'mnist_cnn_v3':
        return create_compile_mnist_cnn_v3(input_shape, num_classes, epochs, lr)
    elif name == 'mnist_cnn_v4':
        return create_compile_mnist_cnn_v4(input_shape, num_classes, epochs, lr)
    elif name == 'cifar_mlp':
        return create_compile_cifar_mlp(input_shape, num_classes)
    elif name == 'cifar_cnn_v1':
        return create_compile_cifar_cnn_v1(input_shape, num_classes, epochs, lr)
    elif name == 'cifar_cnn_v2':
        return create_compile_cifar_cnn_v2(input_shape, num_classes, epochs, lr)
    elif name == 'cinic_v1':
        return create_compile_cinic_v1(input_shape, num_classes, epochs, lr)
    elif name == 'cinic_v2':
        return create_compile_cinic_v2(input_shape, num_classes, epochs, lr)
    elif name == 'cinic_v3':
        return create_compile_cinic_v3(input_shape, num_classes, epochs, lr)
    else:
        return None


# Creates and compiles an MLP - our baseline model
def create_compile_mnist_mlp(num_pixels, num_classes):
    model = Sequential()
    model.add(Dense(num_pixels,
                    input_dim=num_pixels,
                    kernel_initializer='normal',
                    activation='relu')
              )
    model.add(Dense(num_classes,
                    kernel_initializer='normal',
                    activation='softmax')
              )

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    return model


def create_compile_cifar_mlp(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(np.prod(input_shape),
                    kernel_initializer='normal',
                    activation='relu')
              )
    model.add(Dense(num_classes,
                    kernel_initializer='normal',
                    activation='softmax')
              )

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    return model

# Create a basic convolutional neural netwok - our CNN baseline
def create_compile_mnist_cnn_v1(input_shape, num_classes, epochs, lr):
    model = Sequential()

    model.add(Conv2D(32, (5, 5),
                     input_shape=input_shape,
                     activation='relu')
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    return model


# A step up from the baseline CNN model
# This is an example of an increment in complexity that
# we might want to go about when we face under-fitting
# Notice this model has 2 pairs of convolutional/pooling layers
# and a more complex "MLP-like" portion of the network
def create_compile_mnist_cnn_v2(input_shape, num_classes, epochs, lr):
    model = Sequential()
    model.add(Conv2D(30, (5, 5),
                     input_shape=input_shape,
                     activation='relu')
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3),
                     activation='relu')
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    return model


# An improved version of the cnn to use in the mnist/fashion-mnist datasets
def create_compile_mnist_cnn_v3(input_shape, num_classes, epochs, lr):
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     input_shape=input_shape,
                     activation='relu')
              )
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3),
                     activation='relu')
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3),
                     activation='relu')
              )
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3),
                     activation='relu')
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    return model

# An improved version of the cnn to use in the mnist/fashion-mnist datasets
def create_compile_mnist_cnn_v4(input_shape, num_classes, epochs, lr):
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     input_shape=input_shape,
                     activation='relu')
              )
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3),
                     activation='relu')
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3),
                     activation='relu')
              )
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3),
                     activation='relu')
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    return model


# Creates and compiles a basic cnn to be used as a classifier with
# the cifar dataset. This is our baseline model for that case study
def create_compile_cifar_cnn_v1(input_shape, num_classes, epochs, lr):
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     input_shape=input_shape,
                     padding='same',
                     activation='relu',
                     kernel_constraint=maxnorm(3)
                     )
              )
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_constraint=maxnorm(3))
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512,
                    activation='relu',
                    kernel_constraint=maxnorm(3))
              )
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr=lr,
              momentum=0.9,
              decay=1e-06,
              nesterov=False
              )
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
    return model


# Creates and compiles a more advanced model to use with the
# CIFAR10 dataset.
def create_compile_cifar_cnn_v2(input_shape, num_classes, epochs, lr):
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     input_shape=input_shape,
                     activation='relu',
                     padding='same'
                     )
              )
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    sgd = SGD(lr=lr,
              momentum=0.9,
              decay=1e-06,
              nesterov=False
              )
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
    return model


# Creates and compiles a model to be used in the cinic-10 dataset
def create_compile_cinic_v1(input_shape, num_classes, epochs, lr):
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     input_shape=input_shape,
                     activation='relu',
                     padding='same'
                     )
              )
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    sgd = SGD(lr=lr,
              momentum=0.9,
              decay=1e-06,
              nesterov=False
              )
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
    return model


def create_compile_cinic_v2(input_shape, num_classes, epochs, lr):
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     input_shape=input_shape,
                     activation='relu',
                     padding='same'
                     )
              )
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    sgd = SGD(lr=lr,
              momentum=0.9,
              decay=1e-06,
              nesterov=False
              )
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
    return model


def create_compile_cinic_v3(input_shape, num_classes, epochs, lr):
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     input_shape=input_shape,
                     activation='relu',
                     padding='same'
                     )
              )
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    return model


# Returns a model with the activations of Conv2D and MaxPooling2D layers of a model
def layer_outputs_model(model):
    names = []
    output_layers = []
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, MaxPooling2D):
            output_layers.append(layer.output)
            names.append(layer.name)

    return Model(inputs=model.input, outputs=output_layers), names
