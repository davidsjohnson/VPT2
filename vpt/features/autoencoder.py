from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

import os
import numpy as np


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
        from keras.callbacks import TensorBoard

        self.autoencoder.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


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
### TESTING
##########################

def generate_dataset(fs):

    from skimage.transform import rescale
    import vpt.utils.image_processing as ip

    i_gen = fs.img_generator()

    X = []
    for img, fname in i_gen:

        img = rescale(img, .25, preserve_range=True)
        img = ip.normalize(img)

        X.append(np.expand_dims(img, axis=2))

    return np.array(X)



if __name__ == "__main__":


    # TODO:  Create Pipeline for loading an encoding and using for SFO
    # TODO:     Create Model with more Data /  more epochspip


    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from vpt.streams.file_stream import FileStream

    folder = "data/posture/p4"
    fs = FileStream(folder, ftype='bin')

    X = generate_dataset(fs)
    X_train, X_test = train_test_split(X, test_size=.2)


    # from keras.datasets import mnist
    # (X_train, _), (X_test, _) = mnist.load_data()
    #
    # X_train = X_train.astype('float32') / 255.
    # X_test = X_test.astype('float32') / 255.
    # X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    # X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    #
    # X_train = X_train[:1000]
    # X_test = X_test[:1000]

    img_shape = X_train[0].shape
    print ("Img Shape:", img_shape)
    print ("X Test Shape:", X_test.shape)

    cae = CAE(img_shape=img_shape)
    cae.fit(X_train, X_test, epochs=150, batch_size=200)

    try:
        cae.save("data/cae")
    except Exception as e:
        print ("Issue Saving: ", e)

    encoded_imgs = cae.encode_imgs(X_test)
    decoded_imgs = cae.autoencode_imgs(X_test)

    print ("Encoding Shape:", encoded_imgs.shape)

    n = 10
    plt.figure(figsize=(20, 4))
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


    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):

        # display encoding
        ax = plt.subplot(1, n, i +1)
        plt.imshow(encoded_imgs[i].reshape(encoded_imgs.shape[1] * encoded_imgs.shape[2], encoded_imgs.shape[3]).T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()