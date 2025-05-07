import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


class SVM:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.b = 0  # 初始化为0
        self.support_vectors = None
        self.support_vector_labels = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.unique_labels = le.classes_

        # Create the kernel matrix
        K = self._kernel_matrix(X)

        # Use quadratic programming to optimize
        for _ in range(100):  # Max iterations
            for j in range(n_samples):
                if (y_encoded[j] * (np.sum(self.alpha * y_encoded * K[:, j]) + self.b) < 1):
                    self.alpha[j] += self.C * (1 - y_encoded[j] * (np.sum(self.alpha * y_encoded * K[:, j]) + self.b))

        # 计算偏置b
        self.b = np.mean(y_encoded - np.sum(self.alpha * y_encoded * K.T, axis=1))

        # Store support vectors
        support_vector_indices = self.alpha > 1e-5
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y_encoded[support_vector_indices]
        self.alpha = self.alpha[support_vector_indices]

    def predict(self, X):
        # 计算测试数据与支持向量之间的核矩阵
        K = self._kernel_matrix(X, self.support_vectors)  # 只与支持向量计算
        return np.sign(np.sum(self.alpha * self.support_vector_labels * K, axis=1) + self.b)

    def _kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            K = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                for j in range(X2.shape[0]):
                    K[i, j] = np.exp(-np.linalg.norm(X1[i] - X2[j]) ** 2)
            return K
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")


# 从Excel导入鸢尾花数据集
data = pd.read_excel('E:\study\机器学习\LAB\lab2\word\iris_data.xlsx')  # 确保Excel文件在同一目录下
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values  # 标签

# 选择前两个特征
X = X[:, :2]  # 选择第0和第1特征（前两个特征）

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练SVM
svm = SVM(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)

# 进行预测
predictions = svm.predict(X_test)

# 打印精度
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')


# 可视化超平面（适用于2D数据）
def plot_decision_boundary(svm, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)

    # 使用LabelEncoder编码的标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 绘制训练数据的散点图
    plt.scatter(X[:, 0], X[:, 1], c=y_encoded, edgecolors='k', marker='o')
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# 可视化训练数据的决策边界
plot_decision_boundary(svm, X_train, y_train)
