import numpy as np
import os
import nrrd
import random


def gen_file_list(file_path):
    return sorted(list(os.listdir(file_path)))


def random_shuffle(*lists):
    idx = list(zip(*lists))
    random.shuffle(idx)
    return zip(*idx)


def batch_generator(file_path, mask_path, batch_size):
    file_list = gen_file_list(file_path)
    mask_list = gen_file_list(mask_path)

    while True:
        file_list, mask_list = random_shuffle(file_list, mask_list)
        for (file, mask) in zip(file_list, mask_list):

            X = load_data(file, file_path)
            Y = load_data(mask, mask_path)
            X, Y = random_shuffle(X, Y)
            X = np.array(X)
            Y = np.array(Y)
            for i in range(X.shape[0] // batch_size):
                x_batch = X[i * batch_size:(i + 1) * batch_size, ...]
                y_batch = Y[i * batch_size:(i + 1) * batch_size, ...]

                yield x_batch, y_batch


def load_data(data, data_path):
    data, header = nrrd.read(os.path.join(data_path, data))

    data = np.array(data).transpose(2, 1, 0)  # nrrd read/write order shenanigans
    data = (data - np.min(data)) / np.ptp(data)
    data = data[:, :, :, np.newaxis]

    return data
