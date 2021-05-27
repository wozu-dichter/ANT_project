import os
from time import localtime, strftime
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import copy


def get_current_time():
    return strftime("%Y-%m-%d-%H-%M-%S", localtime())


def load_npy(path):
    try:
        data = np.load(path, allow_pickle=True).item()
    except ValueError:
        data = np.load(path, allow_pickle=True)
    return data


def save_npy(path, data):
    np.save(path, data)


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def tf_initialize():
    import tensorflow as tf
    v = tf.__version__
    if v[0] == "1":
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)


def plot_in_figure(fig_name, signal, wait=False):
    plt.figure(fig_name)
    plt.clf()
    plt.plot(signal)
    plt.show()
    if wait:
        plt.waitforbuttonpress()


def custom_sort(my_list):
    def convert(text):
        return float(text) if text.isdigit() else text

    def alphanum(key):
        return [convert(c) for c in re.split("([-+]?[0-9]?[0-9]*)", key)]

    my_list.sort(key=alphanum)
    return my_list


def glob_sorted(path):
    return custom_sort(glob(path))


def deepcopy(x):
    return copy.deepcopy(x)


def plt_show_full_screen():
    # this function must be called before plt.show()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())


def create_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)


tf_initialize()
