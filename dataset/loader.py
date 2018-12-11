# coding=utf-8
from keras.datasets import mnist
import keras
import numpy as np
import h5py


def mnist_preprocessing(imgs):
    """
    Normalize the imgs
    :param imgs:
    :return:
    """
    if imgs.ndim < 4:
        imgs = np.array([imgs])
    imgs = imgs.astype('float32')
    imgs /= 255
    return imgs


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    x_train = mnist_preprocessing(x_train)
    x_test = mnist_preprocessing(x_test)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def load_usps():
    with h5py.File('../dataset/data/usps.h5', 'r') as hf:
        train = hf.get('train')
        X_train = train.get('data')[:]
        Y_train = train.get('target')[:]
        test = hf.get('test')
        X_test = test.get('data')[:]
        Y_test = test.get('target')[:]
    x_train = np.zeros(shape=(X_train.shape[0], 1, 28, 28))
    x_test = np.zeros(shape=(X_test.shape[0], 1, 28, 28))
    for i in np.arange(0, X_train.shape[0]):
        x = X_train[i].reshape(16, 16)
        x = np.pad(x, ((6, 6), (6, 6)), 'constant', constant_values=(0, 0)).reshape((1, 28, 28))
        x_train[i] = x
    for i in np.arange(0, X_test.shape[0]):
        x = X_test[i].reshape(16, 16)
        x = np.pad(x, ((6, 6), (6, 6)), 'constant', constant_values=(0, 0)).reshape((1, 28, 28))
        x_test[i] = x

    y_train = keras.utils.to_categorical(Y_train, 10)
    y_test = keras.utils.to_categorical(Y_test, 10)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    from torchvision import transforms
    #transform = transforms.ToTensor()
    x_train, y_train, x_test, y_test = load_usps()
    x = x_train[0]
    print(x.shape)
    import torch
    # x = torch.from_numpy(x)
    # print(x.shape)
    x = torch.DoubleTensor(x)
    print(x)



