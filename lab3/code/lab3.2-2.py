import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# 自定义的支持向量回归类
class CustomSVR:
    def __init__(self, C=1.0, epsilon=0.1, gamma='scale'):
        self.C = C  # 惩罚参数
        self.epsilon = epsilon  # epsilon-tube, 允许一定的误差
        self.gamma = gamma  # RBF 核函数的参数
        self.support_ = None  # 支持向量
        self.alpha_ = None  # 拉格朗日乘子
        self.b_ = 0  # 偏置

    # 定义核函数 (RBF 核)
    def rbf_kernel(self, X1, X2):
        if self.gamma == 'scale':
            gamma = 1 / X1.shape[1]  # 用于调整 gamma
        else:
            gamma = self.gamma
        return np.exp(-gamma * cdist(X1, X2, 'sqeuclidean'))

    # 模型训练 (使用SMO算法的简化形式)
    def fit(self, X, y):
        m, n = X.shape
        K = self.rbf_kernel(X, X)  # 计算 RBF 核矩阵
        P = K
        q = -y

        # 初始化拉格朗日乘子
        alpha = np.zeros(m)
        for epoch in range(100):  # 简化SMO算法的迭代过程
            for i in range(m):
                Ei = np.dot(P[i], alpha) - y[i]
                if (y[i] * Ei < -self.epsilon and alpha[i] < self.C) or (y[i] * Ei > self.epsilon and alpha[i] > 0):
                    # 选择第二个拉格朗日乘子
                    j = np.random.randint(0, m)
                    while j == i:
                        j = np.random.randint(0, m)
                    Ej = np.dot(P[j], alpha) - y[j]

                    # 更新alpha[i]和alpha[j]
                    alpha_old_i, alpha_old_j = alpha[i], alpha[j]
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = np.clip(alpha[j], L, H)

                    if abs(alpha[j] - alpha_old_j) < 1e-5:
                        continue

                    alpha[i] += y[i] * y[j] * (alpha_old_j - alpha[j])

        # 计算偏置
        support_idx = (alpha > 0)
        self.support_ = X[support_idx]
        self.alpha_ = alpha[support_idx]
        self.y_support_ = y[support_idx]
        self.b_ = np.mean(self.y_support_ - np.dot(self.alpha_, K[support_idx][:, support_idx]))

    # 预测
    def predict(self, X):
        K = self.rbf_kernel(X, self.support_)
        return np.dot(K, self.alpha_) + self.b_


# 加载波士顿房价数据集
boston = fetch_openml(name="boston", version=1, as_frame=False)

# 选择两个特征进行回归
X = boston.data[:, [5, 12]]  # 选择第6列和第13列的特征
y = boston.target  # 目标值（房价）


# 删除异常值（基于IQR）
def remove_outliers(X, y):
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (y >= lower_bound) & (y <= upper_bound)
    return X[mask], y[mask]


X, y = remove_outliers(X, y)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # y 也需要标准化

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用自定义SVR模型
model = CustomSVR(C=100, gamma='scale')
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 将预测结果和真实结果反标准化
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)

# 计算均方误差
mse = np.mean((y_pred_inv - y_test_inv) ** 2)
print(f"测试集的均方误差: {mse:.2f}")
r2 = r2_score(y_test_inv, y_pred_inv)
print(f"测试集的 R² 值: {r2:.2f}")

# 生成 3D 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制测试集的真实值和预测值
ax.scatter(X_test[:, 0], X_test[:, 1], y_test_inv, color='blue', label='真实值')
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred_inv, color='red', label='预测值')

# 设置轴标签
ax.set_xlabel('Feature 6 (RM)')
ax.set_ylabel('Feature 13 (LSTAT)')
ax.set_zlabel('房价')

plt.title('波士顿房价回归 3D 可视化')
plt.legend()
plt.show()
