from data_utils import untar_dir

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tarfile

# Process images of this size. Note that this differs from the original CIFAR

# image size of 32 x 32. If one alters this number, then the entire model

# architecture will change and any model would need to be retrained.

IMAGE_SIZE = 24



# Global constants describing the CIFAR-10 data set.

NUM_CLASSES = 10

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

BATCH_SIZE = 5
# Global constants describing the CIFAR-10 data set.



# Constants describing the training process.

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.

LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

FILENAME = "cifar-10-batches-bin"
SRC_TAR_DIR = "D:\datasets\cifar-10-binary.tar.gz"
DST_PATH = "D:\datasets"

def _parse_function(dataset):
    label_bytes = 1
    height = 32

    width = 32

    depth = 3

    image_bytes = height * width * depth
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(dataset, tf.uint8)
    label = tf.strided_slice(record_bytes, [0], [label_bytes])
    label = tf.cast(label, tf.int32)
    print("lable:",type(label))
    label = tf.reshape(label, [1])

    # The remaining bytes after the label represent the image, which we reshape

    # from [depth * height * width] to [depth, height, width].

    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [depth, height, width])

    # Convert from [depth, height, width] to [height, width, depth].

    image = tf.transpose(depth_major, [1, 2, 0])
    print(image.shape)
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


def load_data(data_dir, batch_size = 128, num_epochs = 5):
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
    print(dataset.output_shapes)
    print(dataset.output_types)

    #解析从二进制文件中读取的一个元素，转换为（label， image）
    dataset = dataset.map(_parse_function)
    print("DATASET_1", dataset)

    dataset = dataset.shuffle(buffer_size=10000)
    print("DATASET_2", dataset)
    dataset = dataset.batch(batch_size)
    print("DATASET_3", dataset)
    dataset = dataset.repeat(num_epochs)
    print("DATASET_4", dataset)


    iterator = dataset.make_one_shot_iterator()
    #返回的one_element为batch_size个（_labels, _features）
    one_element = iterator.get_next()
    print("one_element", one_element)

    """
    with tf.Session() as sess:

        for _ in range(3):
            #print(sess.run(one_element))
            _labels, _features = sess.run(one_element)
            print(_labels)

        
        try:
            while True:
                print(sess.run(one_element))
        except tf.errors.OutOfRangeError:
            print("end!")
    
    """
    return one_element


def read_data(src_tar, dst_path, file_name, batch_size, num_epochs):
    cifar10_dir = os.path.join(dst_path, file_name)
    print(cifar10_dir)
    path = pathlib.Path(cifar10_dir)
    if (not(path.exists())):
        untar_dir(src_tar, dst_path)
    """
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # As a sanity check, we print out the size of the training and test data.
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    #print(y_train[:2])
    return X_train, y_train, X_test, y_test
    """
    return load_data(cifar10_dir, batch_size, num_epochs)

one_element = read_data(SRC_TAR_DIR, DST_PATH, FILENAME, batch_size=BATCH_SIZE, num_epochs=2)

"""
with tf.Session() as sess:
    for _ in range(3):
        # print(sess.run(one_element))
        _labels, _features = sess.run(one_element)
        print(_labels)
"""

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
        tf.truncated_normal(stddev=stddev, dtype=dtype))
    #是否加入L2正则化惩罚项
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


# We instantiate all variables using tf.get_variable() instead of

# tf.Variable() in order to share variables across multiple GPU training runs.

# If we only ran this model on a single GPU, we could simplify this function

# by replacing all instances of tf.get_variable() with tf.Variable().
def build_model(images):
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

    # local3
    # local3-全连接层，384个节点
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # 把单个样本的特征拼成一个大的列向量，以便我们可以执行单个矩阵乘法
        # images.get_shape().as_list()[0]获取images的batch size
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        print(reshape.get_shape()[1])
        dim = reshape.get_shape()[1].value
        print("dim", dim)

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


# 描述损失函数，往inference图中添加生成损失（loss）所需要的操作（ops）
def loss(logits, labels):
    '''
    将L2Loss添加到所有可训练变量
    添加"Loss" and "Loss/avg"的summary
    ARGS：
    logits：来自inference（）的Logits
    labels：来自distorted_inputs或输入（）的标签.一维张量形状[batch_size]

    返回：
    float类型的损失张量
    '''

    labels = tf.cast(labels, tf.int64)
    # 计算这个batch的平均交叉熵损失
    # 添加一个tf.nn.softmax_cross_entropy_with_logits操作，用来比较inference()函数所输出的logits Tensor与labels
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # 总损失定义为交叉熵损失加上所有的权重衰减项（L2损失）
    return tf.add_n(tf.get_collection('losses'), name='total_loss')




def train(total_loss, global_step):

    """Train CIFAR-10 model.



    Create an optimizer and apply to all trainable variables. Add moving

    average for all trainable variables.



    Args:

        total_loss: Total loss from loss().

        global_step: Integer Variable counting the number of training steps

        processed.

    Returns:

        train_op: op for training.

    """

    # Variables that affect learning rate.

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE

    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)



    # Decay the learning rate exponentially based on the number of steps.
    # 根据步骤数以指数方式衰减学习率
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)

    #tf.summary.scalar('learning_rate', lr)



    # Generate moving averages of all losses and associated summaries.

    loss_averages_op = _add_loss_summaries(total_loss)



    # Compute gradients.

    with tf.control_dependencies([loss_averages_op]):

        opt = tf.train.GradientDescentOptimizer(lr)

        grads = opt.compute_gradients(total_loss)



    # Apply gradients.

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)



    # Add histograms for trainable variables.

    for var in tf.trainable_variables():

        tf.summary.histogram(var.op.name, var)



    # Add histograms for gradients.

    for grad, var in grads:

        if grad is not None:

            tf.summary.histogram(var.op.name + '/gradients', grad)



    # Track the moving averages of all trainable variables.

    variable_averages = tf.train.ExponentialMovingAverage(

        MOVING_AVERAGE_DECAY, global_step)

    with tf.control_dependencies([apply_gradient_op]):

        variables_averages_op = variable_averages.apply(tf.trainable_variables())



    return variables_averages_op


