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
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.unique_labels = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.unique_labels = le.classes_

        # Create the kernel matrix
        K = self._kernel_matrix(X)

        # Use quadratic programming to optimize for each class
        for i in range(len(self.unique_labels)):
            self._optimize_for_class(K, y_encoded, i)

    def _optimize_for_class(self, K, y_encoded, class_index):
        n_samples = len(y_encoded)
        # Create binary labels for the current class
        binary_labels = np.where(y_encoded == class_index, 1, -1)

        for _ in range(100):  # Max iterations
            for j in range(n_samples):
                if (binary_labels[j] * (np.sum(self.alpha * binary_labels * K[:, j]) + self.b) < 1):
                    self.alpha[j] += self.C * (1 - binary_labels[j] * (np.sum(self.alpha * binary_labels * K[:, j]) + self.b))

        # Calculate bias b for the current class
        self.b += np.mean(binary_labels - np.sum(self.alpha * binary_labels * K.T, axis=1))

        # Store support vectors
        support_vector_indices = self.alpha > 1e-5
        if self.support_vectors is None:
            self.support_vectors = []
            self.support_vector_labels = []
            self.alpha = []

        self.support_vectors.append(X[support_vector_indices])
        self.support_vector_labels.append(binary_labels[support_vector_indices])
        self.alpha.append(self.alpha[support_vector_indices])

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.unique_labels)))
        for i in range(len(self.unique_labels)):
            K = self._kernel_matrix(X, self.support_vectors[i])
            scores[:, i] = np.sum(self.alpha[i] * self.support_vector_labels[i] * K, axis=1) + self.b

        return self.unique_labels[np.argmax(scores, axis=1)]

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


# Load the Iris dataset
data = pd.read_excel('E:/study/机器学习/LAB/lab2/word/iris_data.xlsx')
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Labels

# Select the first two features
X = X[:, :2]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM
svm = SVM(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)

# Make predictions
predictions = svm.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')


# Visualize decision boundaries for multi-class classification
def plot_decision_boundary(svm, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)

    # Encode labels for plotting
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Plot training data
    plt.scatter(X[:, 0], X[:, 1], c=y_encoded, edgecolors='k', marker='o')
    plt.title('SVM Decision Boundary for Multi-Class Classification')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# Visualize the decision boundary with training data
plot_decision_boundary(svm, X_train, y_train)
