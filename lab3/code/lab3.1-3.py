import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from itertools import combinations
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 核函数
def linear_kernel(x1, x2): # 线性核
    return np.dot(x1, x2)


def gaussian_kernel(x1, x2, sigma=0.5): # 高斯核
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

# SVM实现
class SVM:
    def __init__(self, kernel=gaussian_kernel, C=1.0):
        self.kernel = kernel
        self.C = C
        self.alphas = None
        self.b = 0
        self.w = None
        self.X = None  # 保存训练数据 X
        self.y = None  # 保存训练标签 y

    def fit(self, X, y):
        self.X = X  # 保存训练数据 X
        self.y = y  # 保存训练标签 y
        n = X.shape[0]
        self.alphas = np.zeros(n)
        K = np.array([[self.kernel(X[i], X[j]) for j in range(n)] for i in range(n)])  # 基于二分类任务计算核矩阵
        passes = 0
        while passes < 5:
            num_changed_alphas = 0
            for i in range(n):
                E_i = self._decision_function(i, K) - y[i]
                if (y[i]*E_i < -0.001 and self.alphas[i] < self.C) or (y[i]*E_i > 0.001 and self.alphas[i] > 0):
                    j = np.random.choice([x for x in range(n) if x != i])
                    E_j = self._decision_function(j, K) - y[j]
                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]
                    L, H = self._compute_L_H(y[i], y[j], alpha_i_old, alpha_j_old)
                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    self.alphas[j] -= y[j] * (E_i - E_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    b1 = self.b - E_i - y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                    self.b = b1 if 0 < self.alphas[i] < self.C else (b2 if 0 < self.alphas[j] < self.C else (b1 + b2) / 2)
                    num_changed_alphas += 1
            passes = passes + 1 if num_changed_alphas == 0 else 0
        self.w = np.dot((self.alphas * y), X)

    def _compute_L_H(self, y_i, y_j, alpha_i_old, alpha_j_old):
        if y_i != y_j:
            return max(0, alpha_j_old - alpha_i_old), min(self.C, self.C + alpha_j_old - alpha_i_old)
        return max(0, alpha_j_old + alpha_i_old - self.C), min(self.C, alpha_j_old + alpha_i_old)

    def _decision_function(self, i, K):
        return np.dot(self.alphas * self.y, K[:, i]) + self.b  # 使用 i-th 样本的核矩阵列

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# 多分类SVM
class MultiClassSVM:
    def __init__(self, kernel=gaussian_kernel, C=1.0):
        self.models = {}
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        classes = np.unique(y)
        for (i, j) in combinations(classes, 2):
            X_ij = X[np.where((y == i) | (y == j))]
            y_ij = y[np.where((y == i) | (y == j))]
            y_ij = np.where(y_ij == i, 1, -1)  # 将二分类标签转化为 -1 和 1
            model = SVM(kernel=self.kernel, C=self.C)
            model.fit(X_ij, y_ij)  # 对这两个类进行训练
            self.models[(i, j)] = model  # 存储模型

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.models)))
        for idx, ((i, j), model) in enumerate(self.models.items()):
            pred = model.predict(X)
            votes[:, idx] = np.where(pred == 1, i, j)
        return mode(votes, axis=1)[0].flatten()

def visualize_results(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格以评估模型
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.predict(xy).reshape(XX.shape)

    ax.contourf(XX, YY, Z, alpha=0.3)
    plt.xlabel('length')
    plt.ylabel('width')
    plt.title('SVM Decision Boundaries on Iris Dataset')
    plt.show()

# 加载数据集
# iris = load_iris()
# X = iris.data[:, :2]  # 只使用前两个特征用于可视化
# y = iris.target
data = pd.read_excel('E:\study\机器学习\LAB\lab2\word\iris_data.xlsx')
n = 1 # 第n个特征
m = 3 # 第m个特征
X = data.iloc[:, [n,m]].values  # 特征列
y = data.iloc[:, -1].values  # 目标列
le = LabelEncoder()  # 对标签进行编码
y = le.fit_transform(y)  # 转换为整数编码
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练多分类SVM
model = MultiClassSVM(kernel=gaussian_kernel, C=1.0)
model.fit(X_train, y_train)

# 预测结果
y_pred = model.predict(X_test)

# 计算并打印预测精度
accuracy = np.mean(y_pred == y_test)
print(f"预测精度: {accuracy * 100:.2f}%")

# 可视化决策边界
visualize_results(X_test, y_pred, model)
