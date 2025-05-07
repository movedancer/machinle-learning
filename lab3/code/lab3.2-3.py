import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.datasets import fetch_openml
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score

# 加载波士顿房价数据集
boston = fetch_openml(name="boston", version=1, as_frame=False)

# 选择两个特征进行回归
X = boston.data[:, [5, 9]]  # 选择第6列和第10列的特征
y = boston.target  # 目标值（房价）

# 删除异常值（基于IQR）
def remove_outliers(X, y):
    # 计算四分位数
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1

    # 定义正常范围为 [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 删除超出这个范围的点
    mask = (y >= lower_bound) & (y <= upper_bound)
    return X[mask], y[mask]

# 移除异常值
X, y = remove_outliers(X, y)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # y 也需要标准化

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建支持向量回归模型（SVR），使用 RBF 核
model = SVR(kernel='rbf', C=10, gamma='auto')
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
ax.scatter(X_test[:, 0], X_test[:, 1], y_test_inv, color='blue', label='real')
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred_inv, color='red', label='predict')

# 创建用于绘制平面的网格
xx, yy = np.meshgrid(np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 20),
                     np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 20))
zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 将预测结果反标准化
zz_inv = scaler_y.inverse_transform(zz)

# 绘制回归平面
ax.plot_surface(xx, yy, zz_inv, color='green', alpha=0.5)

# 设置轴标签
ax.set_xlabel('Feature 6 (RM)')
ax.set_ylabel('Feature 10 (TAX)')
ax.set_zlabel('prise')

plt.title('3D Visualization of Boston Housing Prices Regression')
plt.legend()
plt.show()
