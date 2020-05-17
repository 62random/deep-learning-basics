
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
import numpy as np
from keras.constraints import maxnorm
from keras.optimizers import SGD
import keras.backend as K


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
    elif name == 'autoencoderv1':
        return create_compile_autoencoderv1(input_shape)
    elif name == 'autoencoderv2':
        return create_compile_autoencoderv2(input_shape)
    elif name == 'vae':
        return create_compile_variational_autoencoder(input_shape)
    else:
        return None


# Creates and compiles a basic autoencoder
def create_compile_autoencoderv1(num_pixels):
    # Layers
    inputs = Input(shape=(num_pixels,))
    encoded = Dense(units=32, activation='relu')(inputs)
    decoded = Dense(units=num_pixels, activation='sigmoid')(encoded)

    # Models
    encoder = Model(inputs, encoded)
    autoencoder = Model(inputs, decoded)

    autoencoder.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

    return autoencoder, encoder

# Creates and compiles a slightly more complex autoencoder
def create_compile_autoencoderv2(num_pixels):
    input_image= Input(shape=(num_pixels,))
    encoded1 = Dense(units=128, activation='relu')(input_image)
    encoded2 = Dense(units=64, activation='relu')(encoded1)
    encoded_final = Dense(units=32, activation='relu')(encoded2)
    decoded1 = Dense(units=64, activation='relu')(encoded_final)
    decoded2 = Dense(units=128, activation='relu')(decoded1)
    decoded_final = Dense(units=num_pixels, activation='sigmoid')(decoded2)
     
    encoder = Model(inputs=input_image, outputs=encoded_final) 
    autoencoder = Model(inputs=input_image, outputs=decoded_final)
    
    autoencoder.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
    
    return autoencoder, encoder

#####################################################################################
# Variational autoencoder

# Encoding part of the VAE
def create_encoder(input_shape):
    inputs = Input(shape=input_shape) 
    x = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    t_mean = Dense(2)(x) 
    t_log_var = Dense(2)(x)
    # Output of the created model are the sufficient statistics
    # of the variational distriution q(t|x;phi), mean and log variance.
    encoder = Model(inputs=inputs, outputs=[t_mean, t_log_var], name='encoder')
    return encoder

# Decoding part of the VAE
def create_decoder():
    decoder_input = Input(shape=(2,))
    x = Dense(12544, activation='relu')(decoder_input) # 12544 = 14*14*64
    x = Reshape((14, 14, 64))(x)
    x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    
    decoder = Model(inputs=decoder_input, outputs=x, name='decoder')
    return decoder


# Taken from Keras documentation
def sample(args):
    '''
    Draws samples from a standard normal and scales the samples with
    standard deviation of the variational distribution and shifts them
    by the mean.
    Args: sufficient statistics of the variational distribution.
    Returns: Samples from the variational distribution.
    '''
    t_mean, t_log_var = args
    t_sigma = K.sqrt(K.exp(t_log_var))
    epsilon = K.random_normal(shape=K.shape(t_mean), mean=0., stddev=1.)
    return t_mean + t_sigma * epsilon

# Lambda layers in Keras help you to implement layers 
# or functionality that is not prebuilt 
# and which do not require trainable weights.
# In this function, we use it to create a sampling layer
def create_sampler():
    return Lambda(sample, name='sampler') 

# Wrapper of this set of functions, building
# and compiling a variational autoencoder (VAE)
def create_compile_variational_autoencoder(input_shape):

    inputs = Input(shape=input_shape)

    # Encoder
    encoder = create_encoder(input_shape)
    t_mean, t_log_var = encoder(inputs)
    # Sampling layer
    sampler = create_sampler()
    t = sampler([t_mean, t_log_var])
    # Decoder
    decoder = create_decoder()
    t_decoded = decoder(t)
    vae = Model(inputs, t_decoded, name='vae')
    
    
    # To be used as a loss function
    # Taken from keras documentation
    # We have to define this function here because
    # we need the statistics yielded by the sampler
    # (t_mean and t_log_var)
    def neg_variational_lower_bound(input_image, t_decoded):
        '''
        Negative variational lower bound used as loss function
        for training the variational auto-encoder.
        Args:  input_image: input images
               t_decoded: reconstructed images
        '''
        # Reconstruction loss
        rc_loss = K.sum(K.binary_crossentropy(K.batch_flatten(input_image), K.batch_flatten(t_decoded)), axis=-1)
        # Regularization term (KL divergence)
        kl_loss = -0.5 * K.sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=-1)
        # Average over mini-batch
        return K.mean(rc_loss + kl_loss)
    
    # Compile model
    vae.compile(optimizer='rmsprop', 
                loss=neg_variational_lower_bound, 
                metrics=['accuracy'])

    return vae, t_mean, t_log_var, encoder



######################################################################################


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
