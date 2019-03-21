import numpy as np
from keras import backend as K
import scipy.io


def load_NMNIST_test(file_path):

    mat = scipy.io.loadmat(file_path)

    noisy_x_test_unformatted = mat['test_x']
    noisy_y_test = mat['test_y']

    noisy_x_test = []

    for i in range(len(noisy_x_test_unformatted)):
        formatted_row = np.reshape(noisy_x_test_unformatted[i], (28, 28))
        noisy_x_test.append(formatted_row)

    noisy_x_test = np.array(noisy_x_test)
    noisy_y_test = np.array(np.argmax(noisy_y_test, axis=1))

    if K.image_data_format() == 'channels_first':
        noisy_x_test = noisy_x_test.reshape(noisy_x_test.shape[0], 1, 28, 28)
    else:
        noisy_x_test = noisy_x_test.reshape(noisy_x_test.shape[0], 28, 28, 1)


    return noisy_x_test, noisy_y_test




def load_MNIST(file_path):

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255


    return x_train, y_train, x_test, y_test