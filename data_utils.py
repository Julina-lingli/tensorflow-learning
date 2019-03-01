import os
import tarfile
import platform
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf


# Process images of this size. Note that this differs from the original CIFAR

# image size of 32 x 32. If one alters this number, then the entire model

# architecture will change and any model would need to be retrained.

IMAGE_SIZE = 24



# Global constants describing the CIFAR-10 data set.

NUM_CLASSES = 10

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000



def untar_dir(src, dstPath):
    tarHandle = tarfile.open(src, "r:gz")
    for filename in tarHandle.getnames():
        print (filename)
    tarHandle.extractall(dstPath)
    tarHandle.close()

#untar_dir("D:\datasets\cifar-10-python.tar.gz", "../datasets")

def load_pickle(filename):
    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(filename)
    elif version[0] == '3':
        return  pickle.load(filename, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

#加载一个batch
def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    """
    使用pickle模块读取filename内的二进制内容
    """
    with open(filename, 'rb') as f:
        #从“文件”中，读取字符串，将它们反序列化转换为Python的数据对象
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        #transpose后shape为（10000，32，32，3）
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(filename):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(filename, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        #将5个batch加入到一个列表中
        xs.append(X)
        ys.append(Y)

    #将5个batch（10000，32，32，3）拼接为一个（50000，32，32，3）
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(filename, 'test_batch'))
    return Xtr, Ytr, Xte, Yte





