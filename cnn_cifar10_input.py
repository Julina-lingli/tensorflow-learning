from data_utils import untar_dir

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tarfile
import time
from datetime import datetime

# Process images of this size. Note that this differs from the original CIFAR

# image size of 32 x 32. If one alters this number, then the entire model

# architecture will change and any model would need to be retrained.

IMAGE_SIZE = 24



# Global constants describing the CIFAR-10 data set.

NUM_CLASSES = 10

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
#一般取值为64，128，256，512，1024
BATCH_SIZE = 256
# BATCH_SIZE = 1
# Global constants describing the CIFAR-10 data set.
NUM_EPOCHS = 2000
MAX_STEPS = NUM_EPOCHS * (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE)

# Constants describing the training process.BATCH_SIZE

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

NUM_EPOCHS_PER_DECAY = 10.0      # Epochs after which learning rate decays.

LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

FILENAME = "cifar-10-batches-bin"
SRC_TAR_DIR = "D:\datasets\cifar-10-binary.tar.gz"
DST_PATH = "D:\datasets"
TRAIN_DIR = "D:\datasets\cifar10_train"




def _parse_function_train(dataset):
    label_bytes = 1
    height = 32

    width = 32

    depth = 3

    image_bytes = height * width * depth
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(dataset, tf.uint8)
    label = tf.strided_slice(record_bytes, [0], [label_bytes])
    label = tf.cast(label, tf.int32)
    # print("lable:",type(label))
    label = tf.reshape(label, [1])

    # The remaining bytes after the label represent the image, which we reshape

    # from [depth * height * width] to [depth, height, width].

    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [depth, height, width])

    # Convert from [depth, height, width] to [height, width, depth].

    image = tf.transpose(depth_major, [1, 2, 0])
    #print(image.shape)
    reshaped_image = tf.cast(image, tf.float32)

    reshaped_height = IMAGE_SIZE

    reshaped_width = IMAGE_SIZE

    # Image processing for training the network. Note the many random

    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    # 用于训练网络的图像处理，请注意应用于图像的许多随机失真
    # 随机裁剪图像的[height, width]部分
    distorted_image = tf.random_crop(reshaped_image, [reshaped_height, reshaped_width, depth])

    # Randomly flip the image horizontally.
    # 随机水平翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    # 由于这些操作是不可交换的，因此可以考虑随机化和调整操作的顺序
    # 在某范围随机调整图片亮度
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    # 在某范围随机调整图片对比度
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    # 减去平均值并除以像素的方差，白化操作：均值变为0，方差变为1
    distorted_image = tf.image.per_image_standardization(distorted_image)

    return label, distorted_image


def _parse_function_test(dataset):
    label_bytes = 1
    height = 32

    width = 32

    depth = 3

    image_bytes = height * width * depth
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(dataset, tf.uint8)
    label = tf.strided_slice(record_bytes, [0], [label_bytes])
    label = tf.cast(label, tf.int32)

    label = tf.reshape(label, [1])

    # The remaining bytes after the label represent the image, which we reshape

    # from [depth * height * width] to [depth, height, width].

    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [depth, height, width])

    # Convert from [depth, height, width] to [height, width, depth].

    image = tf.transpose(depth_major, [1, 2, 0])
    #print(image.shape)
    reshaped_image = tf.cast(image, tf.float32)

    reshaped_height = IMAGE_SIZE

    reshaped_width = IMAGE_SIZE

    # Image processing for training the network. Note the many random

    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    # 用于训练网络的图像处理，请注意应用于图像的许多随机失真
    # 随机裁剪图像的[height, width]部分
    distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, reshaped_height, reshaped_width)

    # Subtract off the mean and divide by the variance of the pixels.
    # 减去平均值并除以像素的方差，白化操作：均值变为0，方差变为1
    distorted_image = tf.image.per_image_standardization(distorted_image)

    return label, distorted_image
    # return label, image

def load_data_train(data_dir, batch_size = 128, num_epochs = 5):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]

    label_bytes = 1  # 2 for CIFAR-100

    height = 32

    width = 32

    depth = 3

    image_bytes = height * width * depth

    # Every record consists of a label followed by the image, with a

    # fixed number of bytes for each.

    record_bytes = label_bytes + image_bytes
    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    #print("train data:", dataset.output_shapes)
    # print("train data:", dataset.output_types)

    #解析从二进制文件中读取的一个元素，转换为（label， image）
    dataset = dataset.map(_parse_function_train)
    print("DATASET_1", dataset)

    #dataset = dataset.shuffle(buffer_size=100)
    #print("DATASET_2", dataset)
    dataset = dataset.batch(batch_size)
    print("DATASET_3", dataset)
    #对repeat方法不设置重复次数,就不用算repeat的次数
    dataset = dataset.repeat(num_epochs)
    #dataset = dataset.repeat()
    print("DATASET_4", dataset)


    iterator = dataset.make_one_shot_iterator()
    #返回的one_element为batch_size个（_labels, _features）
    next_label, next_feature = iterator.get_next()
    print("next_label", next_label)
    print("next_feature", next_feature)


    """
    
    one_element = iterator.get_next()
    print("one_element", one_element)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Get images and labels for CIFAR-10.

        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on

        # GPU and resulting in a slow down.
        i = 0
        with tf.device('/cpu:0'):
            try:
                #while i < 640:
                while True:
                    _labels, _features = sess.run(one_element)
                    i += 1
            except tf.errors.OutOfRangeError:
                print("load data end!")
            _labels = _labels.flatten()
        print("labels: ", _labels.shape)
        print("_features:", _features.shape)
    print("load_data graph:", tf.get_default_graph())

    # 在可视化器中显示训练图像
    tf.summary.image('images', _features)

    return _features, _labels
    """
    return next_feature, next_label

def load_data_test(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'test_batch.bin')]

    label_bytes = 1  # 2 for CIFAR-100

    height = 32

    width = 32

    depth = 3

    image_bytes = height * width * depth

    # Every record consists of a label followed by the image, with a

    # fixed number of bytes for each.

    record_bytes = label_bytes + image_bytes
    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    # print("test data:", dataset.output_shapes)
    # print("test data:", dataset.output_types)

    # 解析从二进制文件中读取的一个元素，转换为（label， image）
    dataset = dataset.map(_parse_function_train)
    print("DATASET_1", dataset)

    dataset = dataset.batch(batch_size)
    print("DATASET_3", dataset)
    """

    dataset = dataset.shuffle(buffer_size=10000)
    print("DATASET_2", dataset)
    dataset = dataset.batch(batch_size)
    print("DATASET_3", dataset)
    # 对repeat方法不设置重复次数,就不用算repeat的次数
    # dataset = dataset.repeat(num_epochs)
    dataset = dataset.repeat()
    print("DATASET_4", dataset)
    """
    iterator = dataset.make_one_shot_iterator()
    # 返回的one_element为batch_size个（_labels, _features）
    next_label, next_feature = iterator.get_next()
    print("next_label", next_label)
    print("next_feature", next_feature)

    """
    one_element = iterator.get_next()
    print("one_element", one_element)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Get images and labels for CIFAR-10.

        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on

        # GPU and resulting in a slow down.
        i = 0
        with tf.device('/cpu:0'):
            try:
               #while i < 50:
               while True:
                    _labels, _features = sess.run(one_element)
                    i += 1
            except tf.errors.OutOfRangeError:
                print("load data end!")
            _labels = _labels.flatten()
        print("labels: ", _labels.shape)
        print("_features:", _features.shape)
    print("load_data graph:", tf.get_default_graph())

    # 在可视化器中显示训练图像
    tf.summary.image('images', _features)

    return _features, _labels
    """
    return next_feature, next_label

def read_train(src_tar, dst_path, file_name, batch_size, num_epochs):
    cifar10_dir = os.path.join(dst_path, file_name)
    print(cifar10_dir)
    path = pathlib.Path(cifar10_dir)
    if (not(path.exists())):
        untar_dir(src_tar, dst_path)

    X_train, y_train = load_data_train(cifar10_dir, batch_size, num_epochs)

    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)

    return X_train, y_train

def read_test(src_tar, dst_path, file_name, batch_size):
    cifar10_dir = os.path.join(dst_path, file_name)
    print(cifar10_dir)
    path = pathlib.Path(cifar10_dir)
    if (not(path.exists())):
        untar_dir(src_tar, dst_path)

    X_test, y_test = load_data_test(cifar10_dir, batch_size)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    return X_test, y_test
#one_element = read_data(SRC_TAR_DIR, DST_PATH, FILENAME, batch_size=BATCH_SIZE, num_epochs=2)


def load_CIFAR10(X, y):
    """ load all of cifar """
    xs = []
    ys = []
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        i = 0
        try:
             while i < 10:
            # while True:
                _features, _labels = sess.run([X, y])
                i += 1
                _labels = _labels.flatten()
                plt.imshow(_features[0])
                plt.show()
                # 将5个batch加入到一个列表中
                xs.append(_features)
                ys.append(_labels)
                print(i)
        except tf.errors.OutOfRangeError:
            print("load data end!")


    #将5个batch（10000，32，32，3）拼接为一个（50000，32，32，3）
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del _labels, _features
    print(Xtr.shape, Ytr.shape)
    return Xtr, Ytr

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def visualize_cifar10(classes, X_train, y_train):
    num_classes = len(classes)
    samples_per_class = 7
    print(list(enumerate(classes)))
    for y, cls in enumerate(classes):
        #返回从y_train中找到label为y的所有index组成的数组idxs
        idxs = np.flatnonzero(y_train == y)
        print("class:{0} num:{1}".format((y, cls), len(idxs)))
        #从上面返回的idxs数组中随机选择个数为samples_per_class且不重复的成员
        idxs = np.random.choice(idxs, samples_per_class, replace=False)

        #print("idxs:",idxs)
        for i, idx in enumerate(idxs):
            #计算图片显示的位置索引值，比如显示7行10列的图片，第2列图片的位置索引为（2，12，22，32，42，52，62）
            plt_idx = i * num_classes + y + 1
            #print("plt_idx:", plt_idx)
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

def visualize():
    X_train, y_train = read_test(SRC_TAR_DIR, DST_PATH, FILENAME, BATCH_SIZE)
    # """

    
    # X_train, y_train = read_train(SRC_TAR_DIR, DST_PATH, FILENAME, batch_size=BATCH_SIZE, num_epochs=1)
                                              
    # """
    print(X_train, y_train)
    Xtr, Ytr = load_CIFAR10(X_train, y_train)

    visualize_cifar10(classes, Xtr, Ytr)



def _variable_on_cpu(name, shape, initializer):

    """Helper to create a Variable stored on CPU memory.
    Args:

        name: name of the variable

        shape: list of ints

        initializer: initializer for Variable



    Returns:

        Variable Tensor

    """

    with tf.device('/cpu:0'):

        dtype = tf.float32
        # 我们使用tf.get_variable（）而不是tf.Variable（）来实例化所有变量，以便跨多个GPU训练时能共享变量
        # 如果我们只在单个GPU上运行此模型，我们可以通过用tf.Variable（）替换tf.get_variable（）的所有实例来简化此功能
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

    return var


"""
weight decay（权值衰减）使用的目的是防止过拟合。
在损失函数中，weight decay是放在正则项（regularization）前面的一个系数(即超参)，
正则项一般指示模型的复杂度，
所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。
"""
def _variable_with_weight_decay(name, shape, stddev, wd):

    """Helper to create an initialized Variable with weight decay.



    Note that the Variable is initialized with a truncated normal distribution.

    A weight decay is added only if one is specified.



    Args:

    name: name of the variable

    shape: list of ints

    stddev: standard deviation of a truncated Gaussian

    wd: add L2Loss weight decay multiplied by this float. If None, weight

        decay is not added for this Variable.



    Returns:

        Variable Tensor

    """

    dtype = tf.float32
    """
    从截断的正态分布中输出随机值。 
    生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
    """
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    #是否加入L2正则化惩罚项
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


# We instantiate all variables using tf.get_variable() instead of

# tf.Variable() in order to share variables across multiple GPU training runs.

# If we only ran this model on a single GPU, we could simplify this function

# by replacing all instances of tf.get_variable() with tf.Variable().
def build_model(X, X_graph):
    """
    # 读取image
    with tf.Session(graph=X_graph) as X_sess:
        images = X_sess.run(X)
        print("build_model read data graph:", tf.get_default_graph())
    """
    images = tf.reshape(X, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
    print("images:", images.shape)

    print("build_model graph:", tf.get_default_graph())
    # conv1
    # 每一层都创建于一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都将带有其前缀
    with tf.variable_scope('conv1') as scope:
        #权重weights初始化，从截断的正态分布中输出随机值
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=None)
        # 5*5 的卷积核，64个
        # 卷积操作，步长为1，0padding SAME，不改变宽高，通道数变为64
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')

        # 在CPU上创建第一层卷积操作的偏置变量
        #通常将偏置初始化为0，这是因为随机小数值权重矩阵已经打破了对称性。
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))

        # 卷积的结果加上偏置
        pre_activation = tf.nn.bias_add(conv, biases)

        # relu非线性激活
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

        #_activation_summary(conv1)

    # pool1
    # 3*3 最大池化，步长为2
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],

                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,

                      name='norm1')

    # conv2

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',

                                             shape=[5, 5, 64, 64],

                                             stddev=5e-2,

                                             wd=None)

        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')

        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))

        pre_activation = tf.nn.bias_add(conv, biases)

        conv2 = tf.nn.relu(pre_activation, name=scope.name)

        #_activation_summary(conv2)

    # norm2

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,

                      name='norm2')

    # pool2

    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],

                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print("pool2:", pool2)
    # local3
    # local3-全连接层，384个节点
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # 把单个样本的特征拼成一个大的列向量，以便我们可以执行单个矩阵乘法
        # images.get_shape().as_list()[0]获取images的batch size
        #reshape = tf.reshape(pool2, [images.shape[0], -1])
        #reshape = tf.reshape(pool2, [-1, -1])
        dim = 1
        for i in pool2.get_shape().as_list()[1:]:
            dim = dim * i
        print("dim", dim)
        reshape = tf.reshape(pool2, [-1, dim])
        #reshape = tf.reshape(pool2, [tf.shape(images)[0], dim])
        # reshape = tf.reshape(pool2, [tf.shape(images)[0], -1])
        #reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        # print(reshape.get_shape()[1])
        print("reshape", reshape)
        # dim = tf.shape(reshape)[1]
        # dim = reshape.get_shape()[1].value


        #生成该层的weights，并加入L2正则化惩罚wd=0.004
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)

        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))

        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        #_activation_summary(local3)

    # local4

    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],

                                              stddev=0.04, wd=0.004)

        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))

        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

        #_activation_summary(local4)

    # linear layer(WX + b),

    # We don't apply softmax here because

    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits

    # and performs the softmax internally for efficiency.

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],

                                              stddev=1 / 192.0, wd=None)

        biases = _variable_on_cpu('biases', [NUM_CLASSES],

                                  tf.constant_initializer(0.0))

        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

        #_activation_summary(softmax_linear)

    return softmax_linear




#visualize()