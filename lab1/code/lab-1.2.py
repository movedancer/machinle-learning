import numpy as np
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# 导入鸢尾花数据集
iris = load_iris()
x = iris.data  # 输入特征
y = iris.target  # 标签

# 数据集划分，划分为训练集和测试集
def dataset_split(data, labels, test_size = None):
    if isinstance(test_size, float):
        test_size = int(len(data) * test_size)
    shuffled_indices = np.random.permutation(len(data))
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    x_train = data[train_indices]
    x_test = data[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = dataset_split(x, y, test_size=0.3)


# 自定义梯度下降的逻辑回归模型
#2.多分类求解：鸢尾花数据集
class multi_LogisticRegressionGD:
    #初始化函数
    def __init__(self, learning_rate = None, max_iter = None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
    #定义激活函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    #训练方法
    def fit(self, X, y):
        #样本数量，样本特征数量
        num_samples, num_features = X.shape
        #样本种类数量
        num_classes = len(np.unique(y))

        # 初始化参数
        #1.模型权重
        class_weights = np.zeros((num_classes, num_features))
        #2.偏置参数
        class_biases = np.zeros(num_classes)

        for class_label in range(num_classes):
            y_class = np.where(y == class_label, 1, 0)  # 将该类标签设置为1，其他为0
            weights = np.zeros(num_features)
            bias = 0

            for _ in range(self.max_iter):
                linear_model = np.dot(X, weights) + bias
                y_predicted = self.sigmoid(linear_model)

                dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y_class))
                db = (1 / num_samples) * np.sum(y_predicted - y_class)

                weights -= self.learning_rate * dw
                bias -= self.learning_rate * db

            # 保存该类的权重和偏置
            class_weights[class_label] = weights
            class_biases[class_label] = bias

        self.weights = class_weights
        self.bias = class_biases

    def predict(self, X):
        linear_model = np.dot(X, self.weights.T) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = np.argmax(y_predicted, axis=1)  # 选出概率最大的类别
        return y_predicted_class

model = multi_LogisticRegressionGD(learning_rate=0.01, max_iter=2000)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# 对训练集进行预测并计算准确率
train_predictions = model.predict(x_train)
train_accuracy = metrics.accuracy_score(y_train, train_predictions)
print("Logistic Regression模型训练集的准确率：%.3f" % train_accuracy)

# 对测试集进行预测并计算准确率
test_predictions = model.predict(x_test)
test_accuracy = metrics.accuracy_score(y_test, test_predictions)
print("Logistic Regression模型测试集的准确率：%.3f" % test_accuracy)

# 计算正确率
accuracy = metrics.accuracy_score(y_test, predictions)
print("Logistic Regression模型模型正确率：%.3f" % accuracy)

# 输出分类报告
target_names = iris.target_names
print(metrics.classification_report(y_test, predictions, target_names=target_names))