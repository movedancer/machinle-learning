
import numpy as np
import pandas as pd
from collections import Counter
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DecisionTree:
    def __init__(self, method='pre_pruning', min_samples_split=2, max_depth=None):
        self.method = method  # 剪枝方法，'pre_pruning' 或 'post_pruning'
        self.min_samples_split = min_samples_split  # 预剪枝条件
        self.max_depth = max_depth  # 树的最大深度
        self.tree = None

    def fit(self, X, y):
        # 递归构建树
        data = np.c_[X, y]  # 将特征和标签合并
        self.tree = self._build_tree(data, depth=0)
        if self.method == 'post_pruning':
            self._post_pruning(self.tree)

    def _entropy(self, y):
        y = y.astype(int)  # 确保标签是整数类型
        counts = np.bincount(y)  # 计算每个标签的出现次数
        probabilities = counts / len(y)
        return -np.sum([p * log2(p) for p in probabilities if p > 0])

    def _information_gain(self, left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self._entropy(left) - (1 - p) * self._entropy(right)

    def _best_split(self, data):
        best_gain = 0  # 最佳信息增益
        best_question = None  # 最佳分裂点
        current_uncertainty = self._entropy(data[:, -1])
        n_features = data.shape[1] - 1  # 特征数
        for col in range(n_features):
            values = set(data[:, col])  # 该列的所有唯一值
            for val in values:
                left, right = self._partition(data, col, val)
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = self._information_gain(left[:, -1], right[:, -1], current_uncertainty)
                if gain > best_gain:
                    best_gain, best_question = gain, (col, val)
        return best_gain, best_question

    def _partition(self, data, col, val):
        if isinstance(val, (int, float)):  # 判断是数值型特征
            left = data[data[:, col] >= val]
            right = data[data[:, col] < val]
        else:  # 判断是类别型特征
            left = data[data[:, col] == val]
            right = data[data[:, col] != val]
        return left, right

    def _build_tree(self, data, depth):
        X, y = data[:, :-1], data[:, -1]
        if len(set(y)) == 1:
            return y[0]  # 返回单一类标签
        if len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]  # 返回出现次数最多的类
        gain, question = self._best_split(data)
        if gain == 0:
            return Counter(y).most_common(1)[0][0]
        col, val = question
        left, right = self._partition(data, col, val)
        node = {'question': question, 'left': None, 'right': None}
        node['left'] = self._build_tree(left, depth + 1)
        node['right'] = self._build_tree(right, depth + 1)
        return node

    def _post_pruning(self, node):
        if isinstance(node, dict):
            if isinstance(node['left'], dict):
                self._post_pruning(node['left'])
            if isinstance(node['right'], dict):
                self._post_pruning(node['right'])
            if not isinstance(node['left'], dict) and not isinstance(node['right'], dict):
                if node['left'] == node['right']:
                    node = node['left']
        return node

    def predict(self, X):
        return [self._predict_one(row, self.tree) for row in X]

    def _predict_one(self, row, node):
        if not isinstance(node, dict):
            return node
        col, val = node['question']
        if isinstance(val, (int, float)):
            if row[col] >= val:
                return self._predict_one(row, node['left'])
            else:
                return self._predict_one(row, node['right'])
        else:
            if row[col] == val:
                return self._predict_one(row, node['left'])
            else:
                return self._predict_one(row, node['right'])

    def print_tree(self, node=None, spacing=""):
        if node is None:
            node = self.tree
        if not isinstance(node, dict):
            print(spacing + str(node))
            return
        col, val = node['question']
        print(spacing + f"Is feature[{col}] >= {val}?")
        print(spacing + '--> True:')
        self.print_tree(node['left'], spacing + "  ")
        print(spacing + '--> False:')
        self.print_tree(node['right'], spacing + "  ")

def load_dataset(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, :-1].values  # 特征部分
    y = data.iloc[:, -1].values   # 标签部分
    le = LabelEncoder()  # 对标签进行编码
    y = le.fit_transform(y)  # 转换为整数编码
    return X, y

# 数据集加载示例
file_path = 'E:\study\机器学习\LAB\lab2\word\iris_data.xlsx'
X, y = load_dataset(file_path)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树对象，选择预剪枝或后剪枝
tree = DecisionTree(method='pre_pruning', min_samples_split=5, max_depth=5)

# 训练决策树
tree.fit(X_train, y_train)

# 打印训练好的树
tree.print_tree()

# 预测并打印测试集结果
y_pred = tree.predict(X_test)
print("Predictions:", y_pred)

# 打印训练集和测试集的形状
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
