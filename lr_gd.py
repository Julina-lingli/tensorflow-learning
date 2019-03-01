import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import math

# Step 1: read in data from the .xls file
DATA_FILE_XLS = 'data/fire_theft.xls'
DATA_FILE_CVS = 'data/fire_theft.cvs'
data = pd.read_excel(DATA_FILE_XLS, index_col=0)
data.to_csv(DATA_FILE_CVS, encoding='utf-8')
data = pd.read_csv(DATA_FILE_CVS)

print(data.describe())
n_samples = data.count()
#print(("n_samples:", n_samples))
print("data shape:", data.shape)
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
x = np.array(x)
y = np.array(y)
plt.hist(x)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
print("train shape:", X_train.shape)
print("test shape:", X_test.shape)
print("y train shape:", y_train.shape)
train_samples = X_train.shape[0]
x_dim = X_train.shape[1]
#print(train_samples)
print(type(X_train))
print(type(y_train))
#print(X_train.head())
#print(y_train.head())
#X_train = np.array(X_train)
#y_train = np.array(y_train)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print("y shape:", y_train.shape)
#print(type(X_train))
#print(type(y_train))
# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float64, shape=[None, x_dim], name="X")
Y = tf.placeholder(tf.float64, shape=[None, 1], name="Y")

# Step 3: create weight and bias, initialized to 0
#w = tf.Variable(0.0, name="weights")
#b = tf.Variable(0.0, name="bias")
w = tf.Variable(np.random.random([x_dim, 1]), dtype=tf.float64, name="weights")
b = tf.Variable(0.0, dtype=tf.float64, name="bias")

# Step 4: build model to predict Y,设置线性模型y=Wx+b
#Y_predicted = X * w + b
Y_predicted = tf.matmul(X, w) + b

# Step 5: use the square error as the loss function, or mse(mean squared error)
#loss = tf.square(Y - Y_predicted, name="loss")
loss = tf.reduce_mean(tf.square(Y - Y_predicted), name="loss")

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables , w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # Step 8: train the model
    #最大迭代次数
    maxInter = 1000
    tol = 1.e-6
    steps = 0
    # 创建一个空list用于存放成本函数的变化
    cost_history = []
    prevLoss = np.inf
    diff = np.inf
    # 当损失函数的变动小于阈值或达到最大循环次数，则停止迭代
    while ((steps < maxInter) & (diff > tol) ):
        #典型的梯度下降，每次更新参数（一次迭代）需要访问所有的训练集一遍
        _, _loss = sess.run([optimizer, loss], feed_dict={X:X_train, Y:y_train})
        cost_history.append(_loss)

        # 计算损失函数的变动
        diff = abs(prevLoss - _loss)

        prevLoss = _loss
        steps += 1
        print("Epoch {0} loss:{1} diff_loss:{2}".format(steps - 1, _loss, diff))
        #print("（w b）:{0} ".format(sess.run([w, b])))
        '''
        #print("w {0}: b {1} init:".format(sess.run([w, b])))
        #print("（w b）init:{0} ".format(sess.run([w, b])))
        for idx in range(X_train.shape[0]):
            _, l = sess.run([optimizer, loss], feed_dict={X:X_train[idx], Y:y_train[idx]})
            print("loss {0}: {1}".format(idx, l))
            total_loss += l
            print("（w b）:{0} ".format(sess.run([w, b])))
            print("Epoch {0}: {1}: {2}".format(i, total_loss, total_loss/train_samples))
            print("11111111111111111111111")
        '''
    #print(tf.get_default_graph().as_graph_def())
    writer.close()

    w, b = sess.run([w, b])
    y_predict = sess.run(tf.matmul(X_train, w) + b)

    #输出最终的W,b和cost值
    print("W_Value: %f" % w,
            "b_Value: %f" % b,
            "cost_Value: %f" % sess.run(loss,feed_dict={X:X_train, Y:y_train}))
    # 使用模型进行预测
    mse, y_test_predict = sess.run([loss, Y_predicted],
                              feed_dict={X:X_test, Y:y_test})

#y_predict = tf.matmul(X_train, w) + b

plt.plot(X_train, y_train, "bo", label="Real data")
plt.plot(X_train, y_predict, "r", label='Predicted data')
plt.legend()
plt.show()

plt.plot(X_test, y_test, "bo", label="test real data")
plt.plot(X_test, y_test_predict, "r", label="test predict data")
plt.legend()
plt.show()

#绘制成本函数cost在100次训练中的变化情况
plt.plot(range(len(cost_history)), cost_history)
plt.axis([0,100,0,np.max(cost_history)])
plt.xlabel("training epochs")
plt.ylabel("cost")
plt.title("cost history")
plt.show()


r2 = sklm.r2_score(y_test, y_test_predict)
r2_adj = r2 - (x_dim - 1)/(y_test.shape[0] - x_dim) * (1 - r2)
print("test mse(Mean Square Error): %f" % mse)
print("test rmse(Root Mean Square Error): %f" % math.sqrt(mse))
print("test Mean Absolute Error: %f" % sklm.mean_absolute_error(y_test, y_test_predict))
print("test Median Absolute Error: %f" % sklm.median_absolute_error(y_test, y_test_predict))
print("test R^2 : %f" % r2)
print("test Adjusted R^2  : %f" % r2_adj)