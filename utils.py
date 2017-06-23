import cPickle
import os

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_values(keys, fname):
    if not os.path.exists(fname):
        return {}
    with open(fname, 'rb') as f:
        d = cPickle.load(f)
        return {k: d[k] for k in keys if k in d}


def dump_values(d, fname):
    if not os.path.exists(fname):
        old = {}
    else:
        with open(fname, 'rb') as f:
            old = cPickle.load(f)
    old.update(d)
    with open(fname, 'wb') as f:
        cPickle.dump(old, f)


def plot_3d(x, y, z1, z2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z1, c='r')
    if z2 is not None:
        ax.scatter(x, y, z2, c='b')
    plt.show()