import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_openml
from sklearn.metrics import r2_score

# 线性核函数
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# 高斯核函数
def gaussian_kernel(x1, x2, sigma=0.5):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

class SVR:
    def __init__(self, kernel=gaussian_kernel, C=10, epsilon=1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.alphas = None
        self.b = 0
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        n = X.shape[0]
        self.alphas = np.zeros(n)
        K = np.array([[self.kernel(X[i], X[j]) for j in range(n)] for i in range(n)])
        passes = 0
        max_passes = 1000
        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(n):
                E_i = self._decision_function(K[i, :]) - y[i]
                if (abs(E_i) > self.epsilon and self.alphas[i] < self.C) or (abs(E_i) < self.epsilon and self.alphas[i] > 0):
                    j = np.random.choice([x for x in range(n) if x != i])
                    E_j = self._decision_function(K[j, :]) - y[j]
                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]
                    L, H = self._compute_L_H(i, j, alpha_i_old, alpha_j_old)
                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    self.alphas[j] -= (E_i - E_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    self.alphas[i] += alpha_j_old - self.alphas[j]
                    b1 = self.b - E_i - self.alphas[i] * K[i, i] - self.alphas[j] * K[i, j]
                    b2 = self.b - E_j - self.alphas[i] * K[i, j] - self.alphas[j] * K[j, j]
                    self.b = (b1 + b2) / 2
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    def _compute_L_H(self, i, j, alpha_i_old, alpha_j_old):
        if self.y[i] != self.y[j]:
            return max(0, alpha_j_old - alpha_i_old), min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            return max(0, alpha_j_old + alpha_i_old - self.C), min(self.C, alpha_j_old + alpha_i_old)

    # 直接使用一维的 K_row 进行计算
    def _decision_function(self, K_row):
        return np.dot(self.alphas, K_row) + self.b

    def predict(self, X):
        return np.array([self._decision_function(np.array([self.kernel(x, xi) for xi in self.X])) for x in X])

# 定义预处理函数，基于Z-score剔除离群点
def remove_outliers_zscore(X, y, threshold=3):
    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], y[mask]

# 加载波士顿房价数据集
boston = fetch_openml(name="boston", version=1, as_frame=False)

# 选择第n和第m个特征列，假设目标列是最后一列（房价）
n, m = 5, 9  # 选择第6列(RM)和第10列(TAX)作为特征
X = boston.data[:, [n, m]]  # 选择特定的特征列
y = boston.target  # 房价是目标列

# 应用预处理函数，剔除离群点
X, y = remove_outliers_zscore(X, y)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练SVR模型
model = SVR(kernel=gaussian_kernel, C=1.0, epsilon=0.1)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算均方误差 (Mean Squared Error, MSE)
mse = np.mean((y_pred - y_test) ** 2)
print(f"测试集的均方误差: {mse:.2f}")
r2 = r2_score(y_test, y_pred)
print(f"测试集的 R² 值: {r2:.2f}")
# 可视化三维图，展示预测结果与超平面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制真实值和预测值的散点图
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='真实值')
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, color='red', label='预测值')

# 创建网格用于绘制超平面
xx, yy = np.meshgrid(np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 20),
                     np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 20))
zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 绘制超平面
ax.plot_surface(xx, yy, zz, color='green', alpha=0.5, rstride=100, cstride=100)

# 设置轴标签
ax.set_xlabel('Feature 6 (RM)')
ax.set_ylabel('Feature 10 (TAX)')
ax.set_zlabel('房价')

plt.title('三维可视化：真实值 vs 预测值')
plt.legend()
plt.show()
