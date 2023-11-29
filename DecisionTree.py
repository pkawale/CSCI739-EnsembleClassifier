import numpy as np

from Node import Node
from concurrent.futures import ThreadPoolExecutor


class DecisionTree:
    def __init__(self, min_samples_split, max_depth=7, mode = "entropy"):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, data, mode='entropy'):
        self.root = self.build_tree(data, 0, mode)

    def build_tree(self, dataset, depth=0, mode='entropy'):
        X, y = dataset[:, :-1], dataset[:, -1].astype(int)
        num_samples, num_features = np.shape(X)

        # Check stopping conditions
        if num_samples < self.min_samples_split or depth >= self.max_depth:
            leaf_value = self.calculate_leaf_value(y)
            return Node(feature_index=None, threshold=None, left=None, right=None, info_gain=None,
                        value=leaf_value)

        # Find the best split
        best_split = self.get_best_split(dataset, num_features, mode=mode)
        if len(best_split)  == 0:
            best_split["info_gain"] = 0
        # Check for info gain
        if best_split["info_gain"] > 0:
            left_subtree = self.build_tree(best_split["dataset_left"], depth + 1, mode)
            right_subtree = self.build_tree(best_split["dataset_right"], depth + 1, mode)
            return Node(feature_index=best_split["feature_index"], threshold=best_split["threshold"],
                        left=left_subtree, right=right_subtree, info_gain=best_split["info_gain"], value=None)
        else:
            # If no info gain, return a leaf node
            leaf_value = self.calculate_leaf_value(y)
            return Node(feature_index=None, threshold=None, left=None, right=None, info_gain=None,
                        value=leaf_value)

    def calculate_leaf_value(self, y):
        y = y.astype(int)
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def get_best_split(self, dataset, num_features, mode="entropy"):
        best_split = {}
        max_info_gain = -float("inf")

        # Ensure we're working with a NumPy array
        dataset = np.array(dataset)

        X, y = dataset[:, :-1], dataset[:, -1].astype(int)
        unique, counts = np.unique(y, return_counts=True)
        if np.max(counts) / np.sum(counts) > 0.9:
            return best_split

        for feature_index in range(num_features):
            # Sort the data along the feature to calculate possible thresholds
            unique_values = np.unique(X[:, feature_index])

            for i in range(1, len(unique_values)):
                # Convert to a larger integer type before addition to prevent overflow
                threshold = (int(unique_values[i]) + int(unique_values[i - 1])) / 2
                data_left, data_right = dataset[X[:, feature_index] < threshold], dataset[~(X[:, feature_index] < threshold)]

                if len(data_left) > 0 and len(data_right) > 0:
                    # Calculate information gain
                    current_info_gain = self.calculate_information_gain(y, data_left[:, -1], data_right[:, -1], mode)
                    if current_info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "info_gain": current_info_gain,
                            "dataset_left": data_left,
                            "dataset_right": data_right
                        }
                        max_info_gain = current_info_gain

        return best_split

    def calculate_information_gain(self, y, y_left, y_right, mode="entropy"):
        if mode == "gini":
            parent_loss = self.calculate_gini(y)
            child_loss = ((len(y_left) / len(y)) * self.calculate_gini(y_left)
                          + (len(y_right) / len(y)) * self.calculate_gini(y_right))
        else:
            parent_loss = self.calculate_entropy(y)
            child_loss = ((len(y_left) / len(y)) * self.calculate_entropy(y_left)
                          + (len(y_right) / len(y)) * self.calculate_entropy(y_right))
        return parent_loss - child_loss

    def calculate_entropy(self, y):
        unique_classes = np.unique(y)
        entropy = 0.0
        for cls in unique_classes:
            p_cls = np.sum(y == cls) / float(len(y))
            entropy -= p_cls * np.log2(p_cls)
        return entropy

    def calculate_gini(self, y):
        unique_classes = np.unique(y)
        impurity = 1.0
        for cls in unique_classes:
            p_cls = np.sum(y == cls) / float(len(y))
            impurity -= p_cls ** 2
        return impurity

    def predict(self, x, tree=None):
        if tree is None:
            tree = self.root
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.predict(x, tree.left)
        else:
            return self.predict(x, tree.right)

    def predict_all(self, X):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            predictions = list(executor.map(self.predict, X))
        return predictions

