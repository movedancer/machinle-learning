import numpy as np
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

#导入西瓜数据集
'''
data = (
    ("青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.46, "是"),
    ("乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.774, 0.376, "是"),
    ("乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.634, 0.264, "是"),
    ("青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.608, 0.318, "是"),
    ("浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.556, 0.215, "是"),
    ("青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.403, 0.237, "是"),
    ("乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 0.481, 0.149, "是"),
    ("乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 0.437, 0.211, "是"),
    ("乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", 0.666, 0.091, "否"),
    ("青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0.243, 0.267, "否"),
    ("浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0.245, 0.057, "否"),
    ("浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0.343, 0.099, "否"),
    ("青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0.639, 0.161, "否"),
    ("浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0.657, 0.198, "否"),
    ("乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.36, 0.37, "否"),
    ("浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0.593, 0.042, "否"),
    ("青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0.719, 0.103, "否")
)
'''
#columns = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "标签"]
data = (
    ( 0.697, 0.460, "是"),
    ( 0.774, 0.376, "是"),
    ( 0.634, 0.264, "是"),
    ( 0.608, 0.318, "是"),
    ( 0.556, 0.215, "是"),
    ( 0.403, 0.237, "是"),
    ( 0.481, 0.149, "是"),
    ( 0.437, 0.211, "是"),
    ( 0.666, 0.091, "否"),
    ( 0.243, 0.267, "否"),
    ( 0.245, 0.057, "否"),
    ( 0.343, 0.099, "否"),
    ( 0.639, 0.161, "否"),
    ( 0.657, 0.198, "否"),
    ( 0.360, 0.370, "否"),
    ( 0.593, 0.042, "否"),
    ( 0.719, 0.103, "否")
)

columns = [ "密度", "含糖率", "标签"]
df = pd.DataFrame(data, columns=columns)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

#独热码标签编码
x_encoded = pd.get_dummies(x)
y_encoded = pd.get_dummies(y).values  
y_encoded_single = np.argmax(y_encoded, axis=1)  #只选取第一位作为标签，0为好瓜，1为坏瓜

# 数据集划分，划分为训练集和测试集
def dataset_split(data, labels, test_size = None):
    if isinstance(test_size, float):
        test_size = int(len(data) * test_size)
    shuffled_indices = np.random.permutation(len(data))
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    x_train = data.iloc[train_indices]
    x_test = data.iloc[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = dataset_split(x_encoded, y_encoded_single, test_size=0.3)

# 特征缩放
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 自定义梯度下降的逻辑回归模型
#1.二分类求解：西瓜数据集
class sigle_LogisticRegressionGD:
    def __init__(self, learning_rate = None, max_iter = None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    # 定义 Sigmoid 函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 模型训练函数
    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 梯度下降法
        for _ in range(self.max_iter):
            # 计算线性模型
            linear_model = np.dot(x, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # 计算梯度
            dw = (1 / num_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # 更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # 预测函数
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        # 通过概率阈值 0.5 来进行二分类
        return np.where(y_predicted > 0.5, 1, 0)

model = sigle_LogisticRegressionGD(learning_rate=0.01, max_iter=1000)
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
print(metrics.classification_report(y_test, predictions, target_names=["0", "1"])) ##0为好瓜，1为坏瓜