
C++ Implementation details:

#### Test file 
- Test.cc file has a new implementation of Decision Tree from scratch.
- Tested it with creating the **complete** dataset using [RandomForest1.cc](/src/RandomForest1.cc) file. Works but gives 62% accuracy.
- Had tested with the complete dataset in [Python's](./randomForestClassifier.py) implementation and got 95-97% accuracy.


#### DecisionTree
- init/constructor
- fit
- build_tree
- calculate_leaf_value
- get_best_split
- calculate_information_gain
- calculate_entropy
- calculate_gini
- predict
- predict_all

#### Utility functions
- CalculateAccuracy
- Predict
- TrainTestSplit
