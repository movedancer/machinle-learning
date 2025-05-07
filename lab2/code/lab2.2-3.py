import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # 用于评估回归模型

'''

    def _best_split(self, data):
        best_sse = float('inf')  # 最佳误差平方和
        best_question = None  # 最佳分裂点
        current_sse = np.sum((data[:, -1] - np.mean(data[:, -1])) ** 2)  # 当前误差平方和
        n_features = data.shape[1] - 1  # 特征数
        for col in range(n_features):
            values = set(data[:, col])  # 该列的所有唯一值
            for val in values:
                left, right = self._partition(data, col, val)
                if len(left) == 0 or len(right) == 0:
                    continue
                # 计算分裂后的误差平方和
                sse_left = np.sum((left[:, -1] - np.mean(left[:, -1])) ** 2)
                sse_right = np.sum((right[:, -1] - np.mean(right[:, -1])) ** 2)
                # 总的误差平方和
                total_sse = sse_left + sse_right
                if total_sse < best_sse:
                    best_sse, best_question = total_sse, (col, val)
        return best_sse, best_question
        
        
    def _best_split(self, data):
        best_mse = float('inf')  # 最佳均方误差
        best_question = None  # 最佳分裂点
        current_mse = self._mse(data[:, -1])
        n_features = data.shape[1] - 1  # 特征数
        for col in range(n_features):
            values = set(data[:, col])  # 该列的所有唯一值
            for val in values:
                left, right = self._partition(data, col, val)
                if len(left) == 0 or len(right) == 0:
                    continue
                # 计算分裂后的均方误差
                mse_left = self._mse(left[:, -1])
                mse_right = self._mse(right[:, -1])
                # 加权平均均方误差
                weighted_mse = (len(left) * mse_left + len(right) * mse_right) / (len(left) + len(right))
                if weighted_mse < best_mse:
                    best_mse, best_question = weighted_mse, (col, val)
        return best_mse, best_question    
'''

class RegressionTree:
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

    def _mse(self, y):
        """计算均方误差"""
        return np.mean((y - np.mean(y)) ** 2)

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
        if len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)  # 返回均值作为叶子节点
        best_mse, question = self._best_split(data)
        if best_mse == float('inf'):
            return np.mean(y)
        col, val = question
        left, right = self._partition(data, col, val)
        node = {'question': question, 'left': None, 'right': None}
        node['left'] = self._build_tree(left, depth + 1)
        node['right'] = self._build_tree(right, depth + 1)
        return node

    def _best_split(self, data):
        best_sse = float('inf')  # 最佳误差平方和
        best_question = None  # 最佳分裂点
        current_sse = np.sum((data[:, -1] - np.mean(data[:, -1])) ** 2)  # 当前误差平方和
        n_features = data.shape[1] - 1  # 特征数
        for col in range(n_features):
            values = set(data[:, col])  # 该列的所有唯一值
            for val in values:
                left, right = self._partition(data, col, val)
                if len(left) == 0 or len(right) == 0:
                    continue
                # 计算分裂后的误差平方和
                sse_left = np.sum((left[:, -1] - np.mean(left[:, -1])) ** 2)
                sse_right = np.sum((right[:, -1] - np.mean(right[:, -1])) ** 2)
                # 总的误差平方和
                total_sse = sse_left + sse_right
                if total_sse < best_sse:
                    best_sse, best_question = total_sse, (col, val)
        return best_sse, best_question

    def _post_pruning(self, node):
        if isinstance(node, dict):
            if isinstance(node['left'], dict):
                self._post_pruning(node['left'])
            if isinstance(node['right'], dict):
                self._post_pruning(node['right'])
            if not isinstance(node['left'], dict) and not isinstance(node['right'], dict):
                # 计算剪枝前后的均方误差
                left_mean = node['left']
                right_mean = node['right']
                current_mse = self._mse(np.array([left_mean, right_mean]))
                if current_mse >= self._mse(np.array([np.mean([left_mean, right_mean])])):
                    node = np.mean([left_mean, right_mean])  # 剪枝
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
    return X, y

# 数据集加载示例
file_path = 'E:\study\机器学习\LAB\lab2\word\housing_data.xlsx'
#file_path = 'E:\study\机器学习\LAB\lab2\word\icecream_data.xlsx'
X, y = load_dataset(file_path)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建回归树对象，选择预剪枝或后剪枝
tree = RegressionTree(method='post_pruning', min_samples_split=5, max_depth=4)

# 训练回归树
tree.fit(X_train, y_train)

# 打印训练好的树
tree.print_tree()

# 预测训练集和测试集结果
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

# 打印训练集的均方误差和 R² 分数
print("均方误差（MSE）：", mean_squared_error(y_train, y_train_pred))
print("R平方值（R2 Score）：", r2_score(y_train, y_train_pred))

# 打印测试集的均方误差和 R² 分数
print("均方误差（MSE）：", mean_squared_error(y_test, y_test_pred))
print("R平方值（R2 Score）：", r2_score(y_test, y_test_pred))

# 打印训练集和测试集的形状
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)