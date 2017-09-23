from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

import os
import warnings
import numpy as np

import sys
sys.path.append("./")

class CAE:

    def __init__(self, img_shape):

        input_img = Input(shape=img_shape)

        x = Conv2D(100, (5, 5), activation='tanh', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(150, (5, 5), activation='tanh', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(200, (3, 3), activation='tanh', padding="same")(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        print ("shape of encoded", K.int_shape(encoded))

        self.encoder = Model(input_img, encoded)

        x = Conv2D(200, (5, 5), activation='tanh', padding='same')(encoded)
        x = UpSampling2D((2, 2,))(x)
        x = Conv2D(150, (5, 5), activation='tanh', padding='same')(x)
        x = UpSampling2D((2, 2,))(x)
        x = Conv2D(100, (3, 3), activation='tanh', padding="same")(x)
        x = UpSampling2D((2, 2,))(x)
        decoded = Conv2D(1, (3, 3), activation='tanh', padding='same')(x)

        self.autoencoder = Model(input_img, decoded)

        # decoded_layer = self.autoencoder.layers[-1]
        # self.decoder = Model(encoded, decoded_layer(encoded))

        self.autoencoder.compile(optimizer='sgd', loss='mean_squared_error')


    def fit(self, X_train, X_test, epochs=25, batch_size=50):

        self.autoencoder.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_test, X_test))


    def encode_imgs(self, X):
        return self.encoder.predict(X)


    # def decode_imgs(self, encoded):
    #     return self.decoder.predict(encoded)


    def autoencode_imgs(self, X):
        return self.autoencoder.predict(X)


    def save(self, base_folder):

        self.encoder.save(os.path.join(base_folder, "encoder_model.h5"))
        self.autoencoder.save(os.path.join(base_folder, "cae_model.h5"))



##########################
### Training
##########################

def generate_dataset(fs):

    from skimage.transform import rescale
    import vpt.utils.image_processing as ip

    i_gen = fs.img_generator()

    X = []
    for img, fname in i_gen:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = rescale(img, .50, preserve_range=True)

        img = ip.normalize(img)

        X.append(np.expand_dims(img, axis=2))

    return np.array(X)



if __name__ == "__main__":

    import argparse

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from vpt.streams.file_stream import FileStream

    # Configure Command Line arguments
    parser = argparse.ArgumentParser(description="Train Convolutional Autoencoder for image feature extraction.")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument("-f", "--folder", type=str, help="Folder containing the participant recording", metavar="<video folder>", required=True)
    required.add_argument("-e", "--epochs", type=int, help="The number of epochs the training should run for", metavar="<num epochs>", required=True)
    required.add_argument("-b", "--batchsize", type=int, help="The number of images per batch", metavar="<batch size>", required=True)

    args = parser.parse_args()

    folder = args.folder
    n_epochs = args.epochs
    batch_size = args.batchsize

    # Generate dataset and split it
    fs = FileStream(folder, ftype='bin')
    X = generate_dataset(fs)
    X_train, X_test = train_test_split(X, test_size=.1)

    # Train the Autoencoder
    cae = CAE(img_shape=X_train[0].shape)
    cae.fit(X_train, X_test, epochs=n_epochs, batch_size=batch_size)

    # Save the Model
    try:
        cae.save("data/cae")
    except Exception as e:
        print ("Issue Saving: ", e)

    ## Test and visualize the results
    n = 10
    encoded_imgs = cae.encode_imgs(X_test[:n])
    decoded_imgs = cae.autoencode_imgs(X_test[:n])

    plt.figure()
    for i in range(n):

        print (X_test[i].shape)

        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(X_test[i].reshape((X_test[i].shape[:-1])))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1+ n)
        plt.imshow(decoded_imgs[i].reshape((decoded_imgs[i].shape[:-1])))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.figure()
    for i in range(n):

        # display encoding
        ax = plt.subplot(1, n, i +1)
        plt.imshow(encoded_imgs[i].reshape(encoded_imgs.shape[1] * encoded_imgs.shape[2], encoded_imgs.shape[3]).T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("results.pdf")