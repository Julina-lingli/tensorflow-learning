#use sklearn
import sklearn.model_selection as ms
import numpy as np
import numpy.random as nr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sklm

#load iris_data
'''
from sklearn.datasets import load_iris
load_iris(return_X_y)
参数：
1.return_X_y:若为True,则以(data,target)形式返回数据，
            默认为False,表示以字典形式返回数据全部信息(包含data和target)

data,targer = load_iris(return_X_y = True)
''' \
'''

iris = load_iris()
print(type(iris))
print(iris.data.shape)
print(iris.target.shape)
iris_x = iris.data
iris_y = iris.target
print(iris_x[:2])
print(list(iris.feature_names))
print(iris.target_names)
#print(iris.DESCR)
print(iris_x[:2])
'''
import seaborn as sns
#sns.set()
iris = sns.load_dataset("iris")
print(iris.shape)
print(iris.head())
#sns.pairplot(iris, hue='species', height=2.5)


def plot_iris(iris):
    '''Function to plot iris data by type'''
    setosa = iris[iris['species'] == 'setosa']
    versicolor = iris[iris['species'] == 'versicolor']
    virginica = iris[iris['species'] == 'virginica']
    fig, ax = plt.subplots(2, 2, figsize=(12,12))
    x_ax = ['sepal_length', 'sepal_width']
    y_ax = ['petal_length', 'petal_width']
    for i in range(2):
        for j in range(2):
            ax[i,j].scatter(setosa[x_ax[i]], setosa[y_ax[j]], marker = 'x')
            ax[i,j].scatter(versicolor[x_ax[i]], versicolor[y_ax[j]], marker = 'o')
            ax[i,j].scatter(virginica[x_ax[i]], virginica[y_ax[j]], marker = '+')
            ax[i,j].set_xlabel(x_ax[i])
            ax[i,j].set_ylabel(y_ax[j])
    plt.show()

#plot_iris(iris)

X_iris = iris.drop("species",axis=1)
y_iris = iris["species"]


'''
Creates a numpy array of the features.
Numerically codes the label using a dictionary lookup, and converts it to a numpy array. 
'''
Features = np.array(X_iris)

levels = {'setosa':0, 'versicolor':1, 'virginica':2}
labels = np.array([levels[x] for x in y_iris])
print("labels:", labels.shape)
print(labels[:5])
X_train, X_test, y_train,  y_test = train_test_split(Features, labels,
                                                    test_size=0.3, random_state=1115)

#预处理
scale = preprocessing.StandardScaler().fit(X_train)
X_train = scale.transform(X_train)

#建立模型，训练模型
nr.seed(44)
rf_clf = RandomForestClassifier(n_estimators=5)
print("y_train:")
print(y_train[:5])
rf_clf.fit(X_train, y_train)

#验证模型
X_test = scale.transform(X_test)
scores = rf_clf.predict(X_test)


def print_metrics_3(labels, scores):
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score Setosa   Score Versicolor    Score Virginica')
    print('Actual Setosa      %6d' % conf[0, 0] + '            %5d' % conf[0, 1] + '             %5d' % conf[0, 2])
    print('Actual Versicolor  %6d' % conf[1, 0] + '            %5d' % conf[1, 1] + '             %5d' % conf[1, 2])
    print('Actual Vriginica   %6d' % conf[2, 0] + '            %5d' % conf[2, 1] + '             %5d' % conf[2, 2])
    ## Now compute and display the accuracy and metrics
    print('')
    print('Accuracy        %0.2f' % sklm.accuracy_score(labels, scores))
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    print(' ')
    print('          Setosa  Versicolor  Virginica')
    print('Num case   %0.2f' % metrics[3][0] + '     %0.2f' % metrics[3][1] + '      %0.2f' % metrics[3][2])
    print('Precision   %0.2f' % metrics[0][0] + '      %0.2f' % metrics[0][1] + '       %0.2f' % metrics[0][2])
    print('Recall      %0.2f' % metrics[1][0] + '      %0.2f' % metrics[1][1] + '       %0.2f' % metrics[1][2])
    print('F1          %0.2f' % metrics[2][0] + '      %0.2f' % metrics[2][1] + '       %0.2f' % metrics[2][2])


print_metrics_3(y_test, scores)