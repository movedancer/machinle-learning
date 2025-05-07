import numpy as np

# 计算均方误差
def mse(y):
    if len(y) == 0:
        return 0
    mean_y = np.mean(y)
    return np.mean((y - mean_y) ** 2)

# 划分数据集
def split_dataset(X, y, feature_index, threshold):
    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold
    return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

# 选择最佳划分
def best_split(X, y):
    best_mse = float('inf')
    split_idx, split_threshold = None, None
    n_features = X.shape[1]
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, _, y_left, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            weighted_mse = (len(y_left) / len(y)) * mse(y_left) + (len(y_right) / len(y)) * mse(y_right)
            if weighted_mse < best_mse:
                best_mse = weighted_mse
                split_idx = feature_index
                split_threshold = threshold
    return split_idx, split_threshold

# 定义回归树节点
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# 构建回归树
def build_tree(X, y, depth=0, max_depth=None):
    if len(X) == 0 or (max_depth is not None and depth >= max_depth):
        return Node(value=np.mean(y))
    feature_idx, threshold = best_split(X, y)
    if feature_idx is None:
        return Node(value=np.mean(y))
    X_left, X_right, y_left, y_right = split_dataset(X, y, feature_idx, threshold)
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth)
    return Node(feature_idx, threshold, left_subtree, right_subtree)

# 预测函数
def predict(sample, tree):
    if tree.value is not None:
        return tree.value
    if sample[tree.feature_index] <= tree.threshold:
        return predict(sample, tree.left)
    else:
        return predict(sample, tree.right)

# 批量预测
def predict_batch(X, tree):
    return [predict(sample, tree) for sample in X]

# 打印决策树
def print_tree(node, spacing=""):
    if node.value is not None:
        print(spacing + f"Predict: {node.value:.3f}")
        return
    print(spacing + f"Feature {node.feature_index} <= {node.threshold}")
    print(spacing + '--> Left:')
    print_tree(node.left, spacing + "  ")
    print(spacing + '--> Right:')
    print_tree(node.right, spacing + "  ")

# 示例数据集
X_train = np.array([[2.7, 2.5], [1.0, 1.0], [3.0, 4.0], [1.0, 0.0], [0.0, 1.5]])
y_train = np.array([2.5, 1.2, 3.8, 0.9, 1.7])
X_test = np.array([[2.5, 2.1], [1.0, 0.5]])

# 训练回归树
tree = build_tree(X_train, y_train)

# 打印回归树
print("回归树结构：")
print_tree(tree)

# 训练集预测
train_predictions = predict_batch(X_train, tree)
print("\n训练集预测结果:", train_predictions)

# 测试集预测
test_predictions = predict_batch(X_test, tree)
print("测试集预测结果:", test_predictions)
