import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # 仅用于加载数据集
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
data = load_iris()
X = data.data
y = data.target

# 标准化数据
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 手动实现PCA算法
def pca_manual(X, n_components=2):
    # 计算协方差矩阵
    covariance_matrix = np.cov(X, rowvar=False)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # 对特征值进行排序，选择最大的特征值对应的特征向量
    sorted_indices = np.argsort(eigenvalues)[::-1]
    selected_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    # 转换数据到新空间
    X_pca = np.dot(X, selected_eigenvectors)
    return X_pca

# 手动实现对率回归算法
class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.num_features = X.shape[1]
        self.weights = np.zeros((self.num_features, self.num_classes))
        self.bias = np.zeros((1, self.num_classes))

        # One-hot encoding of y
        y_one_hot = np.zeros((y.size, self.num_classes))
        y_one_hot[np.arange(y.size), y] = 1

        # Gradient Descent
        for i in range(self.num_iterations):
            # Linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply softmax to get probabilities
            y_pred = self.softmax(linear_model)
            # Compute gradients
            dw = np.dot(X.T, (y_pred - y_one_hot)) / y.shape[0]
            db = np.sum(y_pred - y_one_hot, axis=0) / y.shape[0]
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(linear_model)
        return np.argmax(y_pred, axis=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 不使用PCA的对率回归模型
logistic_model_without_pca = LogisticRegressionManual(learning_rate=0.01, num_iterations=5000)
logistic_model_without_pca.fit(X_train, y_train)
y_pred_without_pca = logistic_model_without_pca.predict(X_test)

# 打印直接使用对率回归的准确率
accuracy_without_pca = accuracy_score(y_test, y_pred_without_pca)
print(f"不使用PCA的对率回归分类准确率: {accuracy_without_pca * 100:.2f}%")

# 使用PCA将数据降维到2维
X_pca = pca_manual(X, n_components=2)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# 使用PCA降维后的对率回归模型
logistic_model_with_pca = LogisticRegressionManual(learning_rate=0.01, num_iterations=5000)
logistic_model_with_pca.fit(X_train_pca, y_train_pca)
y_pred_with_pca = logistic_model_with_pca.predict(X_test_pca)

# 打印使用PCA降维后的对率回归准确率
accuracy_with_pca = accuracy_score(y_test_pca, y_pred_with_pca)
print(f"使用PCA降维后的对率回归分类准确率: {accuracy_with_pca * 100:.2f}%")

# 绘制降维前的分类结果示意图（不使用PCA的结果）
plt.figure(figsize=(12, 6))

# 不使用PCA的分类示意图
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50, cmap=plt.cm.Paired)
plt.title('Logistic Regression without PCA')
plt.xlabel('Feature 1: Sepal Length')
plt.ylabel('Feature 2: Sepal Width')

# 使用PCA的分类示意图
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = logistic_model_with_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', s=50, cmap=plt.cm.Paired)
plt.title('Logistic Regression with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()
