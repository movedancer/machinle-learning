import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split


# 计算信息熵
def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


# 计算基尼指数
def gini(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return 1 - np.sum([p ** 2 for p in ps])


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


# 预剪枝条件
def should_pre_prune(y, min_samples_split, max_depth, depth):
    if max_depth is None:
        max_depth = float('inf')
    if len(y) < min_samples_split or depth >= max_depth:
        return True
    return False


# 构建决策树
def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2, criterion='entropy', pre_pruning=False):
    n_samples, n_features = X.shape
    n_labels = len(np.unique(y))

    # 预剪枝
    if n_labels == 1 or n_samples == 0 or (pre_pruning and should_pre_prune(y, min_samples_split, max_depth, depth)):
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)

    feature_idx, threshold = best_split(X, y, criterion)

    if feature_idx is None:
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)

    X_left, X_right, y_left, y_right = split_dataset(X, y, feature_idx, threshold)

    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split, criterion, pre_pruning)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split, criterion, pre_pruning)

    return Node(feature_idx, threshold, left_subtree, right_subtree)


# 后剪枝函数
def post_pruning(node, X, y, validation_data):
    if node.left and node.right:
        if node.left.value is None:
            post_pruning(node.left, *split_dataset(X, y, node.feature_index, node.threshold))
        if node.right.value is None:
            post_pruning(node.right, *split_dataset(X, y, node.feature_index, node.threshold))

        left_value = node.left.value
        right_value = node.right.value

        if left_value is not None and right_value is not None:
            y_true = validation_data[1]
            y_pred_no_prune = predict_batch(validation_data[0], node)
            accuracy_no_prune = np.mean(y_true == y_pred_no_prune)

            node_value = Counter(y).most_common(1)[0][0]
            node.value = node_value
            accuracy_prune = np.mean(y_true == [node_value] * len(y_true))

            if accuracy_prune > accuracy_no_prune:
                node.left = node.right = None


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


# 加载自定义数据集
def load_custom_data(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


# 主函数，选择预剪枝或后剪枝
def main(file_path, test_size=0.2, criterion='entropy', max_depth=None, min_samples_split=2, pre_pruning=False,
         post_pruning_flag=False):
    # 加载数据
    X, y = load_custom_data(file_path)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 构建决策树
    tree = build_tree(X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion,
                      pre_pruning=pre_pruning)

    # 后剪枝
    if post_pruning_flag:
        post_pruning(tree, X_train, y_train, (X_test, y_test))

    # 打印决策树
    print("决策树结构：")
    print_tree(tree)

    # 训练集预测
    train_predictions = predict_batch(X_train, tree)
    print("\n训练集预测结果:", train_predictions)

    # 测试集预测
    test_predictions = predict_batch(X_test, tree)
    print("测试集预测结果:", test_predictions)

# 示例调用
main('E:\study\机器学习\LAB\lab2\word\iris_data.xlsx', pre_pruning=True, max_depth=3)
# main('E:\study\机器学习\LAB\lab2\word\iris_data.xlsx', post_pruning_flag=True)
