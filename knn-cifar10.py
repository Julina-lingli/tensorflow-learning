from data_utils import untar_dir, load_CIFAR10
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


FILENAME = "cifar-10-batches-py"
SRC_TAR_DIR = "D:\datasets\cifar-10-python.tar.gz"
DST_PATH = "D:\datasets"

def read_data(src_tar, dst_path, file_name):
    cifar10_dir = os.path.join(dst_path, file_name)
    print(cifar10_dir)
    path = pathlib.Path(cifar10_dir)
    if (not(path.exists())):
        untar_dir(src_tar, dst_path)
    """
    else:
        # Cleaning up variables to prevent loading data multiple times
        try:
            del X_train, y_train
            del X_test, y_test
            print('Clear previously loaded data.')
        except:
            pass
    """
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # As a sanity check, we print out the size of the training and test data.
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    #print(y_train[:2])
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = read_data(SRC_TAR_DIR, DST_PATH, FILENAME)

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

#visualize_cifar10(classes, X_train, y_train)

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)


def kNearestNeighbor(X_train, y_train, X_test, k = 1):
    input_dim = X_train.shape[1]
    test_num = X_test.shape[0]
    X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
    Y = tf.placeholder(dtype=tf.float32, shape=[1, input_dim])

    distance = tf.reduce_sum(tf.sqrt(tf.pow(X - Y, 2)), axis = 1)

    #获取前k个距离最近的
    #top_k = np.argsort(distance)[:k]
    top_k = tf.nn.top_k(-distance, k)
    predict = np.zeros(test_num)
    #print(X_test[0].shape)
    #print(type(np.asmatrix(X_test[0])))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(test_num):
            min_k_dist = sess.run(top_k, feed_dict={
                X:X_train, Y:np.asmatrix(X_test[i])
            })
            predict[i] = np.argmax(np.bincount(y_train[min_k_dist[1]]))

    return predict

def predict_accuracy(y_predict, y):
    import time
    num_correct = np.sum(y_predict == y)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
    print(time.asctime(time.localtime(time.time())))

    return accuracy

# Let's compare how fast the implementations are
def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

def cross_validation(X_train, y_train, num_folds = 5):

    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []

    X_train_folds = np.array_split(X_train, num_folds, axis=0)  # list
    y_train_folds = np.array_split(y_train, num_folds, axis=0)  # list

    """
    循环5次，设置将第1个array设置为验证集，后4个设置为训练集，训练集通过np.concatenate进行纵向合并！

    这里参考网上大佬的代码，下面来解释一下，主要是里面的swap data，这里截一张k折图，每次将当前的fold作业一个验证集，
    而在这里通过交换验证集的数据，将验证集的数据交换到fold1位置，这样便于处理！

    后面便是通过选择k来进行模型训练，寻找最佳k即可！
    """
    k_to_accuracies = {}
    for i in range(num_folds):
        # train / validation split (80% 20%)
        X_train_batch = np.concatenate(X_train_folds[1:num_folds])
        y_train_batch = np.concatenate(y_train_folds[1:num_folds])
        X_valid_batch = X_train_folds[0]
        y_valid_batch = y_train_folds[0]
        # swap data (for next iteration)
        if i < num_folds - 1:
            tmp = X_train_folds[0]
            X_train_folds[0] = X_train_folds[i + 1]
            X_train_folds[i + 1] = tmp
            tmp = y_train_folds[0]
            y_train_folds[0] = y_train_folds[i + 1]
            y_train_folds[i + 1] = tmp
        # train model
        # compute accuracy for each k
        for k in k_choices:
            y_valid_pred = kNearestNeighbor(X_train_batch, y_train_batch, X_valid_batch, k=k)
            # compute validation accuracy
            accuracy = predict_accuracy(y_valid_pred, y_valid_batch)
            #num_correct = np.sum(y_valid_pred == y_valid_batch)
            #accuracy = float(num_correct) / y_valid_batch.shape[0]
            # accumulate accuracy into dictionary
            if i == 0:
                k_to_accuracies[k] = []
            k_to_accuracies[k].append(accuracy)


    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    # plot the raw observations
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

predict = kNearestNeighbor(X_train, y_train, X_test, k = 5)
predict_accur = predict_accuracy(predict, y_test)
cross_validation(X_train, y_train)

