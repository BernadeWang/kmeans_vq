from time import time

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle


def recreate_img(codebook, labels, w, h):
    img = np.zeros((w, h, codebook.shape[1]))
    label_index = 0
    for i in range(w):
        for j in range(h):
            img[i][j] = codebook[labels[label_index]]
            label_index += 1
    return img


def api_train_vq(img, n_colors=64):
    width, height, depth = img.shape
    img = np.array(img, dtype=np.float64) / 255
    img_array = img.reshape((-1, depth))

    t0 = time()
    sample_num = 1000
    img_sample = shuffle(img_array, random_state=0)[:sample_num]
    model = KMeans(n_clusters=n_colors, n_init=8, max_iter=200, random_state=0).fit(img_sample)
    fit_time = time() - t0
    print('fit used %0.3fs.' % fit_time)

    t1 = time()
    labels = model.predict(img_array)
    predict_time = time() - t1
    print('predict used %0.3fs' % predict_time)

    after = recreate_img(model.cluster_centers_, labels, width, height)
    return img, after
