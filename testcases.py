import time
import sys
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import argparse
from randomForestClassifier import RandomForest
from sklearn.preprocessing import LabelEncoder
from keras.src.datasets import mnist

def preprocess_titanic(df):
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Convert categorical variables to numeric
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    # Select features and target
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    target = 'Survived'
    return df[features], df[target]

def main():
    print(f'Hello Random Forester!')

    # Take user inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of trees", type=int)
    parser.add_argument("-s", help="minimum samples to split", type=int)
    parser.add_argument("-d", help="maximum depth of the tree", type=int)
    parser.add_argument("-f", help="maximum number of features to use in a tree", type=int)
    parser.add_argument("-i", help="information gain type gini/entropy")
    parser.add_argument("-df", help="dataset to be used from available list")

    args = parser.parse_args()

    # Initialize all the parameters
    num_trees, min_samples_to_split, max_depth, max_features, mode, dataset = 100, 3, 10, None, "entropy", "iris"

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
        
    if args.df == 'iris':
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
        # print(f'Random Forest Predictions: {predictions}')
        print(f'Accuracy: {accuracy:.2f}')
    elif args.df == 'titanic':
        start_time = time.time()
        # Load the Titanic dataset
        df = pd.read_csv('data/train.csv')

        # Preprocess the dataset
        X, y = preprocess_titanic(df)

        # Split the DataFrame into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, stratify=y, random_state=42)

        # Create and fit the random forest model
        rf = RandomForest(num_trees=num_trees, min_samples_split=min_samples_to_split, 
                        max_depth=max_depth, max_features=max_features, mode=mode)
        rf.fit(np.column_stack((X_train, y_train)))

        # Predict the responses for the test dataset
        predictions = rf.predict_all(X_test.values)

        # Calculate the accuracy
        accuracy = np.sum(y_test.values == np.array(predictions)) / len(y_test)
        print(f'Random Forest Predictions: {predictions}')
        print(f'Accuracy: {accuracy:.2f}')
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total execution time: {total_time:.2f} seconds")
    elif args.df == 'mnist':
        start_time = time.time()
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        # # Flatten the images and normalize pixel values
        train_X_flattened = train_X.reshape((-1, train_X.shape[1] * train_X.shape[2]))
        test_X_flattened = test_X.reshape((-1, test_X.shape[1] * test_X.shape[2]))

        train_X_flattened = train_X_flattened[:1000]
        train_y = train_y[:1000]
        test_X_flattened = test_X_flattened[:500]
        test_y = test_y[:500]

        # Combine the features and targets for the training set
        train_df = np.column_stack((train_X_flattened, train_y))

        # Create and fit the random forest model
        rf = RandomForest(num_trees=10, min_samples_split=2)
        rf.fit(train_df)

        # Predict the responses for the test dataset
        predictions = rf.predict_all(test_X_flattened)
        #
        # Calculate the accuracy
        accuracy = np.sum(test_y == np.array(predictions)) / len(test_y)
        print(f'Random Forest Predictions: {predictions}')
        print(f'Accuracy: {accuracy:.2f}')
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total execution time: {total_time:.2f} seconds")

    else:
        print('Invalid dataset selected. Please choose between iris, titanic, mnist')
        sys.exit(1)

if __name__ == "__main__":
    main()