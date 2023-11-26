#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <numeric>
#include <set>
#include <float.h>
#include "Node.h"

enum rule_mode{Gini, Entropy};

typedef struct BestSplit
{
    size_t feature_index;
    double threshold;
    double info_gain;
    std::vector<std::vector<double>> left_data;
    std::vector<int> left_label;
    std::vector<std::vector<double>> right_data;
    std::vector<int> right_label;
}split;

template <typename T>
std::pair<std::vector<T>, std::vector<T>> sliceVector(const std::vector<T>& input, size_t index) {
    // Check if the index is within the valid range
    if (index >= input.size()) {
        std::cerr << "Error: Index out of range." << std::endl;
        // You can handle this error in a way that fits your application
        // For simplicity, I'm returning two empty vectors
        return { {}, {} };
    }

    // Create two vectors using the constructor with iterators
    std::vector<T> firstSlice(input.begin(), input.begin() + index);
    std::vector<T> secondSlice(input.begin() + index, input.end());

    // Return the sliced vectors
    return { firstSlice, secondSlice };
}

// Function to get the indices of a sorted column
template <typename T>
std::vector<size_t> sort_indices(const std::vector<std::vector<T>>& vec, size_t col) {
    std::vector<size_t> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0);  // Initialize indices to [0, 1, 2, ..., vec.size()-1]

    // Sort the indices based on the specified column
    std::sort(indices.begin(), indices.end(), [&vec, col](size_t i, size_t j) {
        return vec[i][col] < vec[j][col];
    });

    return indices;
}


class DecisionTree {
private:
    int min_samples_split;
    int max_depth;
    Node* root;

public:
    int treeId=0;
    DecisionTree(int min_samples_split, int max_depth = 7, int id)
        : min_samples_split(min_samples_split), max_depth(max_depth), root(NULL), treeId(id) {}

    Node* buildTree(std::vector<std::vector<double>>& X, std::vector<int>& y, int depth = 0) {
        size_t num_samples = X.size(), num_features = X[0].size();

        // Base case
        if(num_samples < min_samples_split || depth >= max_depth){
            int leaf_val = calculateLeafValue(y);
            Node* temp = new Node(0, 0, NULL, NULL, 0, leaf_val);
            return temp;
        }

        // Find the best split
        split best_split = getBestSplit(X, y, num_features, Entropy);

         Node* temp;
        if(best_split.info_gain > 0){
            Node* left_subtree = buildTree(best_split.left_data, best_split.left_label, depth+1);
            Node* right_subtree = buildTree(best_split.right_data, best_split.right_label, depth+1);

           temp = new Node(best_split.feature_index, best_split.threshold, left_subtree, right_subtree, best_split.info_gain, 0);
        }
        
        // If no info gain, return a leaf node
        else{
            temp = new Node(0, 0, NULL, NULL, 0, calculateLeafValue(y));
        }
        
        return temp;
    }

    void fit(std::vector<std::vector<double>>& X, std::vector<int>& y) {
        root = buildTree(X, y);
    }

    
    /// @brief Finds the maximum occuring class
    /// @param y Class label
    /// @return Class which is occurring the maximum
    int calculateLeafValue(const std::vector<int>& y) {
        std::unordered_map<int, int> class_occurences;
        int max_frequent_class = 0;
        size_t max_freq = -1;
        // Find the class which occurs the most
        for(auto a_class:y){
            if (class_occurences.find(a_class) == class_occurences.end()){
                class_occurences[a_class] = 0;                
            }
            class_occurences[a_class]++;
            if(class_occurences[a_class] > max_freq){
                max_frequent_class = a_class;
                max_freq = class_occurences[a_class];
            }
        }
        std::cout<<"leaf-val"<<max_frequent_class<<std::endl;
        return max_frequent_class;
    }

    /// @brief 
    /// @param y 
    /// @return 
    std::set<int> get_unique_classes(const std::vector<int>& y){
        std::set<int> unique_set;
        for(auto a_class:y)
            unique_set.insert(a_class);

        return unique_set;
    }

    double calculate_entropy(const std::vector<int>& y){
        std::set<int> unique_classes = get_unique_classes(y);
        double entropy = 0.0;

        for(auto a_class:unique_classes){
            double p_cls = 0.0;
            for(auto an_element:y){
                if(an_element == a_class){
                    p_cls += 1;
                }
            }
            p_cls /= y.size();
            entropy -= p_cls * log2(p_cls);
        }

        return entropy;
    }

    double calculate_gini(const std::vector<int> y){

        std::set<int> unique_classes = get_unique_classes(y);

        double impurity = 1.0;

        for(auto a_class:unique_classes){
            double p_cls = 0;
            for(auto an_element:y){
                if(an_element == a_class){
                    p_cls += 1;
                }
            }
            p_cls /= y.size();
            impurity -= (p_cls * p_cls);
        }

        return impurity;
    }

    double calculate_information_gain(const std::vector<int> y, const std::vector<int> y_left, const std::vector<int> y_right, rule_mode mode){

        double parent_loss = 0.0, child_loss = 0.0;

        if(mode == Gini){
            parent_loss = calculate_gini(y);
            child_loss = (y_left.size() / y.size()) * calculate_gini(y_left) + 
                                (y_right.size() / y.size()) * calculate_gini(y_right);
        }
        else{
            parent_loss = calculate_entropy(y);
            child_loss = (y_left.size() / y.size()) * calculate_entropy(y_left) + 
                                (y_right.size() / y.size()) * calculate_entropy(y_right);
        }

        return parent_loss - child_loss;

    }

    split getBestSplit(std::vector<std::vector<double>>& dataset, std::vector<int>& y, size_t num_features, rule_mode mode=Entropy){
        split best_split;

        double max_info_gain = -DBL_MAX;

        for(size_t feature_index=0; feature_index<num_features; ++feature_index){
            // Sort the data along the feature to calculate possible thresholds
            std::vector<size_t> sorted_indices = sort_indices(dataset, feature_index);

            std::vector<std::vector<double>> sorted_X;
            std::vector<double> sorted_y;

            for(auto index:sorted_indices){
                sorted_X.push_back(dataset[index]);
                sorted_y.push_back(y[index]);
            }

            for(size_t i = 1; i < sorted_X.size(); ++i){
                double threshold = (sorted_X[i][feature_index] + sorted_X[i-1][feature_index]) / 2;
                
                auto slicedVectors = sliceVector(dataset, i);
                std::vector<std::vector<double>> data_left = slicedVectors.first, data_right = slicedVectors.second;

                auto sliced_y = sliceVector(y, i);
                std::vector<int> y_left = sliced_y.first, y_right = sliced_y.second;
                
                if(data_left.size() > 0 && data_right.size() > 0){
                    double current_info_gain = calculate_information_gain(y, y_left, y_right, mode);

                    if(current_info_gain > max_info_gain){
                        max_info_gain = current_info_gain;
                        best_split.feature_index = feature_index;
                        best_split.threshold = threshold;
                        best_split.info_gain = current_info_gain;
                        best_split.left_data = data_left;
                        best_split.right_data = data_right;
                        best_split.left_label = y_left;
                        best_split.right_label = y_right;
                    }
                }
            }
        }

        return best_split; 

    }

    int predict(std::vector<double> x, Node* tempTree=NULL){
        if(tempTree == NULL){
            tempTree = root;
        }
        // std::cout<<"Here it comes"<<tempTree->feature_index<<std::endl;
        if(root == NULL || tempTree == NULL){
            std::cout<<"The culprit!";
        }
        if(tempTree->value != 0){
            return tempTree->value;
        }

        double feature_val = x[tempTree->feature_index];
        if(feature_val <= tempTree->threshold)
            return predict(x, tempTree->left);
        else
            return predict(x, tempTree->right);
    }

};