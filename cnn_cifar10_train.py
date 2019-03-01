from data_utils import untar_dir

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tarfile
import time
from datetime import datetime

import cnn_cifar10_input as model_input

# Process images of this size. Note that this differs from the original CIFAR

# image size of 32 x 32. If one alters this number, then the entire model

# architecture will change and any model would need to be retrained.

IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.

NUM_CLASSES = 10

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
# 一般取值为64，128，256，512，1024
BATCH_SIZE = model_input.BATCH_SIZE
# Global constants describing the CIFAR-10 data set.
NUM_EPOCHS = model_input.NUM_EPOCHS

MAX_STEPS = model_input.MAX_STEPS

# Constants describing the training process.BATCH_SIZE

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

NUM_EPOCHS_PER_DECAY = 10.0  # Epochs after which learning rate decays.

LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

FILENAME = "cifar-10-batches-bin"
SRC_TAR_DIR = "D:\datasets\cifar-10-binary.tar.gz"
DST_PATH = "D:\datasets"
TRAIN_DIR = "D:\datasets\cifar10_train"

# 描述损失函数，往inference图中添加生成损失（loss）所需要的操作（ops）
def loss(logits, labels, data_graph):
    '''
    将L2Loss添加到所有可训练变量
    添加"Loss" and "Loss/avg"的summary
    ARGS：
    logits：来自inference（）的Logits
    labels：来自distorted_inputs或输入（）的标签.一维张量形状[batch_size]

    返回：
    float类型的损失张量
    '''
    """
    
    with tf.Session(graph=data_graph) as data_sess:
        labels = data_sess.run(labels)
    """
    # labels = labels.flatten()
    print("labels:", labels)
    labels = tf.reshape(labels, [-1,])
    labels = tf.cast(labels, tf.int64)

    # 计算这个batch的平均交叉熵损失
    # 添加一个tf.nn.softmax_cross_entropy_with_logits操作，用来比较inference()函数所输出的logits Tensor与labels
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # 总损失定义为交叉熵损失加上所有的权重衰减项（L2损失）
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):

    """Add summaries for losses in CIFAR-10 model.



    Generates moving average for all losses and associated summaries for

    visualizing the performance of the network.



     Args:

        total_loss: Total loss from loss().

    Returns:

        loss_averages_op: op for generating moving averages of losses.

    """

    # Compute the moving average of all individual losses and the total loss.
    # 计算所有单个损失和总损失的移动平均
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')

    losses = tf.get_collection('losses')

    loss_averages_op = loss_averages.apply(losses + [total_loss])



    # Attach a scalar summary to all individual losses and the total loss; do the

    # same for the averaged version of the losses.

    for l in losses + [total_loss]:

        # Name each loss as '(raw)' and name the moving average version of the loss

        # as the original loss name.

        tf.summary.scalar(l.op.name + ' (raw)', l)

        tf.summary.scalar(l.op.name, loss_averages.average(l))



    return loss_averages_op


def train(total_loss, global_steps):

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
    #print("decay_steps:", decay_steps)
    # Decay the learning rate exponentially based on the number of steps.
    # 根据步骤数以指数方式衰减学习率
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
        global_steps,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)

    #tf.summary.scalar('learning_rate', lr)



    # Generate moving averages of all losses and associated summaries.

    loss_averages_op = _add_loss_summaries(total_loss)

    # 神经网络训练开始前很难估计所需的迭代次数global_step，系统在训练时会自动更新global_step,
    # 学习速率第一次训练开始变化，global_steps每次自动加1
    with tf.control_dependencies([loss_averages_op]):
        method = tf.train.GradientDescentOptimizer(lr)
        optimizer = method.minimize(total_loss, global_step=global_steps)

    """
    # Compute gradients.

    with tf.control_dependencies([loss_averages_op]):

        opt = tf.train.GradientDescentOptimizer(lr)

        grads = opt.compute_gradients(total_loss)



    # Apply gradients.

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    """


    """
    # Add histograms for trainable variables.

    for var in tf.trainable_variables():

        tf.summary.histogram(var.op.name, var)



    # Add histograms for gradients.

    for grad, var in grads:

        if grad is not None:

            tf.summary.histogram(var.op.name + '/gradients', grad)


    """
    # Track the moving averages of all trainable variables.

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_steps)
    #更新列表中的变量tf.trainable_variables()所有参加训练的变量参数
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([optimizer]):

        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    return variables_averages_op

# def do_train(train_dir, _labels, _features, data_graph):
def do_train(train_dir):
    # 读取数据
    """
    with tf.Graph().as_default() as data_graph:
        _features, _labels = model_input.read_train(SRC_TAR_DIR, DST_PATH, FILENAME,
                                                  batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)
        print("main graph_read_data:", tf.get_default_graph())
    """
    data_graph = 0
    #因为数据读取的计算图在该函数外，而两者需要在同一个计算图中，故不能重新设置计算图
    with tf.Graph().as_default():
        _features, _labels = model_input.read_train(SRC_TAR_DIR, DST_PATH, FILENAME,
                                                    batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)
        print("do_train graph:", tf.get_default_graph())

        #print("_features:", _features.shape)
        #print("_labels:", _labels.shape)
        #global_step = tf.train.get_or_create_global_step()
        step = 0
        global_step = tf.Variable(0, trainable=False)
        # Build a Graph that computes the logits predictions from the

        # inference model.
        logits = model_input.build_model(_features, data_graph)

        # Calculate loss.
        total_loss = loss(logits, _labels, data_graph)

        # Build a Graph that trains the model with one batch of examples and

        # updates the model parameters.

        train_op = train(total_loss, global_step)
        #print(type(train_op))
        # 用tf.train.Saver()创建一个Saver来管理模型中的所有变量
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=tf.ConfigProto(log_device_placement = False)) as sess:
            sess.run(tf.global_variables_initializer())
            # 若batch size为64，则5000个数据集可划分为5000/64个batch，
            # dataset在执行iterator.get_next()时返回一个batch的数据集，
            # 那么多个时期比如100个epoch迭代的数据集总共的batch 个数就为100*（5000/64）
            print("MAX_STEPS: ", MAX_STEPS)
            try:
                while step < MAX_STEPS:
                    # 记录运行计算图一次的时间
                    start_time = time.time()
                    _, _total_loss = sess.run([train_op, total_loss])
                    duration_time = time.time() - start_time
                    step +=1
                    print("step:", step)
                    if step % 10 == 0:
                        num_examples_per_step = BATCH_SIZE
                        examples_per_sec = num_examples_per_step / duration_time
                        sec_per_batch = float(duration_time)

                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(), step, _total_loss,
                                            examples_per_sec, sec_per_batch))
                    """
                    if step % 100 == 0:
                        # 添加summary日志
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                    """
                    # 定期保存模型检查点
                    if step % 100 == 0 or (step + 1) == MAX_STEPS:
                        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
            except tf.errors.OutOfRangeError:
                    print("End of dataset per epoch")






def train_main():
    train_dir = TRAIN_DIR
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)

    tf.gfile.MakeDirs(train_dir)
    # 重置tensorflow的graph，确保神经网络可多次运行
    tf.reset_default_graph()
    tf.set_random_seed(1908)
    do_train(train_dir)

train_main()

