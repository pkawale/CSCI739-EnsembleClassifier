import numpy as np
import pandas as pd
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split
import argparse
from DecisionTree import DecisionTree
from concurrent.futures import ProcessPoolExecutor


class RandomForest:
    def __init__(self, num_trees=100, min_samples_split=2, max_depth=7, max_features=None, mode="entropy"):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.mode = mode

    def fit(self, dataset):
        dataset = np.array(dataset)
        for _ in range(self.num_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            random_subset_data = self._bootstrap_sample(dataset)
            tree.fit(random_subset_data, mode=self.mode)
            self.trees.append(tree)

    def _bootstrap_sample(self, data):
        n_samples = data.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        feature_indices = np.arange(data.shape[1] - 1)  # Use all features for simplicity
        if self.max_features is not None:
            feature_indices = self._get_feature_indices(len(feature_indices))
        subset_data = data[indices][:, feature_indices]
        return np.c_[subset_data, data[indices, -1]]

    def _get_feature_indices(self, n_features):
        feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        return feature_indices

    def predict(self, x):
        tree_preds = np.array([tree.predict(x) for tree in self.trees])
        return np.bincount(tree_preds).argmax()

    def predict_all(self, X):
        with ProcessPoolExecutor() as executor:
            predictions = list(executor.map(self.predict, X))
        return predictions


def main():
    print(f'Hello Random Forester!')

    # Take user inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of trees", type=int)
    parser.add_argument("-s", help="minimum samples to split", type=int)
    parser.add_argument("-d", help="maximum depth of the tree", type=int)
    parser.add_argument("-f", help="maximum number of features to use in a tree", type=int)
    parser.add_argument("-i", help="information gain type gini/entropy")

    args = parser.parse_args()

    # Initialize all the parameters
    num_trees, min_samples_to_split, max_depth, max_features, mode = 100, 3, 10, None, "entropy"

    if args.n:
        if(args.n < 0):
            print(f"Invalid number of trees: {args.n}")
            sys.exit(1)
        num_trees = args.n
    if args.s:
        if(args.s < 0):
            print(f"Invalid number of samples to split: {args.s}")
            sys.exit(1)
        min_samples_to_split = args.s
    if args.d:
        if(args.d < 0):
            print(f"Invalid dpeth of tree: {args.d}")
            sys.exit(1)
        max_depth = args.d
    if args.f:
        if(args.f < 0):
            print(f"Invalid number of features: {args.f}")
            sys.exit(1)
        max_features = args.f
    if args.i:
        if(args.f == 'entropy' or args.f == 'gini'):
            print(f"Invalid tree type: {args.i}")
            sys.exit(1)
        mode = args.i

    # Load the iris dataset as a pandas DataFrame
    iris = datasets.load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])

    # Split the DataFrame into 60% training and 40% testing
    # Added stratification to ensure that the classes are evenly split.
    train_df, test_df = train_test_split(df, test_size=0.40, stratify=df['target'])
    
    if(train_df.shape[0] < min_samples_to_split):
        print(f"Min samples to split {min_samples_to_split} is greater than total samples {train_df.shape[0]}")
        sys.exit(1)
    
    if(max_features != None and train_df.shape[1]-1 < max_features):
        print(f"Max features is greater than the total number of features")
        sys.exit(1)

    # Create and fit the random forest model
    # Slightly increased the max_depth and decreased num_trees for demonstration.
    rf = RandomForest(num_trees=num_trees, min_samples_split=min_samples_to_split, 
                      max_depth=max_depth, max_features=max_features, mode=mode)
    
    rf.fit(train_df.values)

    # Prepare X_test and y_test for accuracy evaluation
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.astype(int)

    # Predict the responses for the test dataset
    predictions = rf.predict_all(X_test)

    # Calculate the accuracy
    accuracy = np.sum(y_test == np.array(predictions)) / len(y_test)
    print(f'Accuracy: {accuracy:.2f}')


if __name__ == "__main__":
    main()