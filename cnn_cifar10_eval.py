import tensorflow as tf
import numpy as np
import cnn_cifar10_input as model_input
# from cnn_cifar10_input import read_test
# from cnn_cifar10_input import build_model
from datetime import datetime
import time

CHECKPOINT_DIR = "D:\datasets\cifar10_train"
FILENAME = "cifar-10-batches-bin"
SRC_TAR_DIR = "D:\datasets\cifar-10-binary.tar.gz"
DST_PATH = "D:\datasets"
TRAIN_DIR = "D:\datasets\cifar10_train"

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
RUN_ONCE = False
BATCH_SIZE = model_input.BATCH_SIZE
EVAL_INTERVAL_SECS = 60 * 2

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

def eval_once(saver, top_k_op):
    """Run Eval once.
    Args:

      saver: Saver.

      summary_writer: Summary writer.

      top_k_op: Top K op.

      summary_op: Summary op.

    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:

            # Restores from checkpoint
            # 加载最新的模型
            saver.restore(sess, ckpt.model_checkpoint_path)

            # Assuming model_checkpoint_path looks something like:

            #   /my-favorite-path/cifar10_train/model.ckpt-0,

            # extract global_step from it.

            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("global_step:", global_step)

        else:

            print('No checkpoint file found')

            return

        true_count = 0

        try:
            while True:
                """
                with tf.Session(graph=graph_test_data) as test_data_sess:
                    _X_test, _y_test = test_data_sess.run([X_test, y_test])
                    _y_test = _y_test.flatten()
                    print("_X_test:", _X_test.shape)
                    print("_y_test:", _y_test.shape)
                    top_k_op = predict_op(_X_test, _y_test)
                """

                #获得一个batch 的测试样本的预测正确的，是一个布尔类型的向量表
                predictions = sess.run([top_k_op])
                #获得一个batch大小的样本中预测正确的样本个数
                true_count += np.sum(predictions)
                print("true_count:", true_count)
        except tf.errors.OutOfRangeError:
            print("End of dataset for test")

        """
        # 获得一个batch 的测试样本的预测正确的，是一个布尔类型的向量表
        predictions = sess.run([top_k_op])
        print("predictions:", len(predictions))
        print(predictions)
        # 获得一个batch大小的样本中预测正确的样本个数
        true_count = np.sum(predictions)
        print("true_count:", true_count)
        """
        # Compute precision @ 1.
        total_sample_count = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        precision = true_count / total_sample_count

        print('%s: precision @ step:%s = %.3f' % (datetime.now(), global_step, precision))

def predict_op(X_test, y_test, graph_test_data):
    """
    with tf.Session(graph=graph_test_data) as sess:
        _y_test = sess.run(y_test)
        _y_test = _y_test.flatten()
    print("_y_test", _y_test.shape)
    """
    _y_test = tf.reshape(y_test, [-1, ])
    # Build a Graph that computes the logits predictions
    predict_logits = model_input.build_model(X_test, graph_test_data)

    # pred返回每个样本预测类型的概率
    pred = tf.nn.softmax(logits=predict_logits, name="pred")
    print("pred:", pred)
    # Calculate predictions.
    # tf.nn.in_top_k选择每个样本预测类型的最大概率，比较该最大概率的索引值是否与标签y_test中的值相匹配，返回布尔型
    top_k_op = tf.nn.in_top_k(pred, _y_test, 1)
    print("top_k_op:", top_k_op)
    return top_k_op

def evaluate_main():
    # with tf.Graph().as_default() as graph_test_data:
    X_test, y_test = model_input.read_test(SRC_TAR_DIR, DST_PATH, FILENAME, BATCH_SIZE)
    graph_test_data = 0
    top_k_op = predict_op(X_test, y_test, graph_test_data)

    # Restore the moving average version of the learned variables for eval.

    variable_averages = tf.train.ExponentialMovingAverage(

        MOVING_AVERAGE_DECAY)

    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    while True:

        eval_once(saver, top_k_op)

        if RUN_ONCE:

            break

        time.sleep(EVAL_INTERVAL_SECS)

with tf.device('/cpu:0'):
    evaluate_main()