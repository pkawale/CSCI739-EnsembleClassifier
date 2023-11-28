import gzip
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from DecisionTree import DecisionTree


class RandomForest:
    def __init__(self, num_trees=100, min_samples_split=2, max_depth=7, max_features=None):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, dataset):
        dataset = np.array(dataset)
        for _ in range(self.num_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            random_subset_data = self._bootstrap_sample(dataset)
            tree.fit(random_subset_data)
            self.trees.append(tree)

    def _bootstrap_sample(self, data):
        n_samples = data.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        feature_indices = self._get_feature_indices(data.shape[1] - 1)
        subset_data = data[indices][:, feature_indices]
        # Append the target variable to the subset of data
        subset_data = np.c_[subset_data, data[indices, -1]]  # -1 is the index for the target variable
        return subset_data

    def _get_feature_indices(self, n_features):
        feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        return feature_indices

    def predict(self, x):
        tree_preds = [tree.predict(x) for tree in self.trees]
        most_common_output = max(set(tree_preds), key=tree_preds.count)
        return most_common_output

    def predict_all(self, X):
        return [self.predict(x) for x in X]


def load_images(file_path):
    with gzip.open(file_path, 'r') as f:
        # magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count * column_count))
        
        return images


def load_labels(file_path):
    with gzip.open(file_path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels
    

def load_data(ip = 'iris'):
    """
    """
    if(ip == 'iris'):
        # Load the iris dataset as a pandas DataFrame
        iris = datasets.load_iris()
        df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])

        # Split the DataFrame into 60% training and 40% testing
        # Added stratification to ensure that the classes are evenly split.
        train_df, test_df = train_test_split(df, test_size=0.40, stratify=df['target'])
        train_df = train_df.values
        test_df = test_df.values

    elif(ip == 'mnist'):
        train_df = np.c_[load_images('./data/train-images-idx3-ubyte.gz'), 
                        load_labels('./data/train-labels-idx1-ubyte.gz')]
        test_df = np.c_[load_images('./data/t10k-images-idx3-ubyte.gz'), 
                        load_labels('./data/t10k-labels-idx1-ubyte.gz')]

    return train_df, test_df


def main():
    print(f'Hello Random Forester!')
    ip = input("Enter the type of data to use: ")

    if(len(ip) == 0):   ip = 'iris'
    
    train_data, test_data = load_data(ip)

    rf = RandomForest(num_trees=20, min_samples_split=3, max_depth=15, max_features=None)
    
    rf.fit(train_data)

    # Prepare X_test and y_test for accuracy evaluation
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # Predict the responses for the test dataset
    predictions = rf.predict_all(X_test)

    # Calculate the accuracy
    accuracy = np.sum(y_test == np.array(predictions)) / len(y_test)
    print(f'Random Forest Predictions: {predictions}')
    print(f'Accuracy: {accuracy:.2f}')


if __name__ == "__main__":
    main()
