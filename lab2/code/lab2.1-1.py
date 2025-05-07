import numpy as np
from collections import Counter

# 计算信息熵
def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

# 计算基尼指数
def gini(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return 1 - np.sum([p**2 for p in ps])

# 计算信息增益
def information_gain(y, x, criterion='entropy'):
    if criterion == 'entropy':
        criterion_func = entropy
    elif criterion == 'gini':
        criterion_func = gini
    else:
        raise ValueError("Criterion not recognized.")
    
    parent_criterion = criterion_func(y)
    values, counts = np.unique(x, return_counts=True)
    weighted_sum = np.sum([(counts[i] / len(y)) * criterion_func(y[x == v]) for i, v in enumerate(values)])
    
    return parent_criterion - weighted_sum

# 数据集划分
def split_dataset(X, y, feature_index, threshold):
    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold
    return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

# 选择最佳划分
def best_split(X, y, criterion='entropy'):
    best_gain = -1
    split_idx, split_threshold = None, None
    n_features = X.shape[1]

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, _, y_left, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            gain = information_gain(y, X[:, feature_index] <= threshold, criterion=criterion)
            
            if gain > best_gain:
                best_gain = gain
                split_idx = feature_index
                split_threshold = threshold
                
    return split_idx, split_threshold

# 定义决策树节点
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# 构建决策树
def build_tree(X, y, depth=0, max_depth=None, criterion='entropy'):
    n_samples, n_features = X.shape
    n_labels = len(np.unique(y))

    if n_labels == 1 or n_samples == 0 or (max_depth is not None and depth >= max_depth):
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)

    feature_idx, threshold = best_split(X, y, criterion)

    if feature_idx is None:
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)

    X_left, X_right, y_left, y_right = split_dataset(X, y, feature_idx, threshold)
    
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth, criterion)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth, criterion)
    
    return Node(feature_idx, threshold, left_subtree, right_subtree)

# 预测单个样本
def predict(sample, tree):
    if tree.value is not None:
        return tree.value
    
    feature_value = sample[tree.feature_index]
    if feature_value <= tree.threshold:
        return predict(sample, tree.left)
    else:
        return predict(sample, tree.right)

# 预测多个样本
def predict_batch(X, tree):
    return [predict(sample, tree) for sample in X]

# 打印决策树
def print_tree(node, spacing=""):
    if node.value is not None:
        print(spacing + "Predict:", node.value)
        return

    print(spacing + f"Feature {node.feature_index} <= {node.threshold}")
    print(spacing + '--> Left:')
    print_tree(node.left, spacing + "  ")

    print(spacing + '--> Right:')
    print_tree(node.right, spacing + "  ")

# 数据集示例
X_train = np.array([[2.7, 2.5],
                    [1.0, 1.0],
                    [3.0, 4.0],
                    [1.0, 0.0],
                    [0.0, 1.5]])

y_train = np.array([0, 0, 1, 1, 0])

X_test = np.array([[2.5, 2.1],
                   [1.0, 0.5]])

# 训练决策树
tree = build_tree(X_train, y_train, criterion='entropy')

# 打印决策树
print("决策树结构：")
print_tree(tree)

# 预测训练集
train_predictions = predict_batch(X_train, tree)
print("\n训练集预测结果:", train_predictions)

# 预测测试集
test_predictions = predict_batch(X_test, tree)
print("测试集预测结果:", test_predictions)
