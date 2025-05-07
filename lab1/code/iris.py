import sklearn
import numpy as np
import optuna
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征
y = iris.target  # 标签

# np.random.seed(34)

indices = np.random.permutation(X.shape[0])

# 使用随机索引打乱 X 和 y
X_shuffled = X[indices]
y_shuffled = y[indices]


X_test = X[-30:]  # 测试集
y_test = y[-30:]  # 测试集标签
X      = X[:-30]  # 训练集
y      = y[:-30]  # 训练集标签


# One-hot 编码标签
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

# 初始化权重和偏置
def initialize_params(num_features, num_classes):
    W = np.random.randn(num_features, num_classes)
    b = np.zeros((1, num_classes))
    return W, b

# Softmax 函数
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 损失函数（交叉熵损失）
def compute_loss(Y, Y_hat):
    m = Y.shape[0]
    # 加入一个极小值以避免 log(0)
    epsilon = 1e-12
    Y_hat = np.clip(Y_hat, epsilon, 1. - epsilon)
    return -np.sum(Y * np.log(Y_hat)) / m


# 梯度下降更新参数
def gradient_descent(X, Y, Y_hat, W, b, learning_rate):
    m = X.shape[0]
    
    dZ = Y_hat - Y
    dW = np.dot(X.T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    
    W -= learning_rate * dW
    b -= learning_rate * db
    
    return W, b

# 训练逻辑回归模型
def train(X, y, num_classes, learning_rate=0.01, num_iterations=1000):
    num_samples, num_features = X.shape
    W, b = initialize_params(num_features, num_classes)
    Y = one_hot_encode(y, num_classes)

    for i in range(num_iterations):
        Z = np.dot(X, W) + b
        Y_hat = softmax(Z)
        loss = compute_loss(Y, Y_hat)
        W, b = gradient_descent(X, Y, Y_hat, W, b, learning_rate)

    return W, b, loss

# 预测函数
def predict(X, W, b):
    Z = np.dot(X, W) + b
    Y_hat = softmax(Z)
    return np.argmax(Y_hat, axis=1)

# 定义目标函数供 Optuna 优化
def objective(trial):
    # 调节超参数
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-1, 1)
    num_iterations = trial.suggest_int('num_iterations', 50, 150)
    
    # 训练模型
    W, b, loss = train(X, y, num_classes=3, learning_rate=learning_rate, num_iterations=num_iterations)
    
    # 使用训练集进行预测，计算准确率作为优化目标
    y_pred = predict(X, W, b)
    accuracy = np.mean(y_pred == y)
    
    # 返回负的准确率作为损失
    return -accuracy

# 使用 Optuna 进行超参数调优
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

# 打印最优超参数
print("Best hyperparameters:", study.best_params)

# 使用最优超参数重新训练模型
best_learning_rate = study.best_params['learning_rate']
best_num_iterations = study.best_params['num_iterations']

W, b, loss = train(X, y, num_classes=3, learning_rate=best_learning_rate, num_iterations=best_num_iterations)
y_pred = predict(X_test, W, b)
accuracy = np.mean(y_pred == y_test)

print(f"Accuracy with best hyperparameters: {accuracy * 100:.2f}%")


