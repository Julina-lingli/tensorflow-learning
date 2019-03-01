import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
import numpy as np
import math
import matplotlib.pyplot as plt


#read data
DATA_FILE_XLS = 'data/fire_theft.xls'
DATA_FILE_CVS = 'data/fire_theft.cvs'
data = pd.read_excel(DATA_FILE_XLS)
#print("data_xls:", type(data))
print(data.columns.values)
poly_data2 = np.square(data["X"])
poly_data3 = np.power(data["X"], 3)
data["X2"] = poly_data2
data["X3"] = poly_data3
#print(data.describe())
print(data.head())

#labels = data.iloc[:, -1]
#datas = data.iloc[:, 0]
labels = data["Y"]
features = data[["X", "X2", "X3"]]
print(features.head())

x_dim = features.shape[1]
print("x_dim:", x_dim)

def datas_hist(df, cols):
    for i in cols:
        plt.hist(df[i],label="i")
        plt.show()
columns = ["X", "X2", "X3"]
datas_hist(features, columns)

#data split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=100)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
print(X_train.head())
#define model
def poly_regression(x_dim, X_train, X_test, y_train, y_test):
    np.random.seed(200)
    X = tf.placeholder(tf.float32, [None, x_dim], name="X")
    Y = tf.placeholder(tf.float32, [None, 1], name="Y")

    w = tf.Variable(np.random.random([x_dim, 1]), dtype=tf.float32, name="weights")
    b = tf.Variable(0.0, dtype=tf.float32, name="bias")

    #define predict func Y_predicted
    Y_predicted = tf.matmul(X, w) + b

    #define loss
    loss = tf.reduce_mean(tf.square(Y - Y_predicted), name="loss")

    #define optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        maxInter = 100
        threshold = 1.e-6
        print("threshold:", threshold)
        steps = 0
        diff = np.inf #初始为无穷大
        pre_loss = np.inf
        loss_history = []
        # 当损失函数的变动小于阈值或达到最大循环次数，则停止迭代
        while((steps < maxInter) and  (diff > threshold)):
            _, _loss = sess.run([optimizer, loss], feed_dict={X:X_train, Y:y_train})

            loss_history.append(_loss)

            diff = abs(pre_loss  - _loss)
            pre_loss = _loss
            steps += 1

            print("steps:{0} loss:{1} diff:{2}".format(steps, _loss, diff))

        print("(w,b):{0} loss:{1}".format(sess.run([w, b]), _loss))
        y_train_predict = sess.run(Y_predicted, feed_dict={X:X_train})
        mse_test, y_test_predict = sess.run([loss, Y_predicted], feed_dict={X:X_test, Y:y_test})

    return y_train_predict, loss_history, mse_test, y_test_predict


def data_plot_legend(X, Y1, Y2, label1, label2):
    plt.plot(X, Y1, "bo", label=label1)
    plt.plot(X, Y2, "r", label=label2)
    plt.legend()
    plt.show()

def loss_plot(loss, epoch):
    plt.plot(range(len(loss)), loss)
    plt.axis([0, epoch, 0, np.max(loss)])
    plt.xlabel("training epochs")
    plt.ylabel("loss")
    plt.show()

def evaluate_model(y, y_predict, x_dim, mse):
    r2 = sklm.r2_score(y, y_predict)
    r2_adj = r2 - (x_dim - 1)/(y_test.shape[0] - x_dim) * (1 - r2)

    print("test mse(Mean Square Error): %f" % mse)
    print("test rmse(Root Mean Square Error): %f" % math.sqrt(mse))
    print("test Mean Absolute Error: %f" % sklm.mean_absolute_error(y, y_predict))
    print("test Median Absolute Error: %f" % sklm.median_absolute_error(y, y_predict))
    print("test R^2 : %f" % r2)
    print("test Adjusted R^2  : %f" % r2_adj)

y_train_predict, loss_history, mse_test, y_test_predict = poly_regression(x_dim, X_train, X_test, y_train, y_test)

data_plot_legend(X_train["X"], y_train, y_train_predict, "train_real", "train_predict")
data_plot_legend(X_test["X"], y_test, y_test_predict, "test_real", "test_predict")

evaluate_model(y_test, y_test_predict, x_dim, mse_test)

