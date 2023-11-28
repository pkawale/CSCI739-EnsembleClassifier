#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
// #include "iris_read.h"

struct Node {
    int feature_index;
    double threshold;
    Node* left;
    Node* right;
    double info_gain;
    int value;
};

class DecisionTree {
public:
    DecisionTree(int min_samples_split, int max_depth = 7) 
        : min_samples_split(min_samples_split), max_depth(max_depth), root(nullptr) {}

    void fit(std::vector<std::vector<double>>& data) {
        root = build_tree(data);
    }

    int predict(std::vector<double>& x, Node* tree = nullptr) {
        if (tree == nullptr) {
            tree = root;
        }

        if (tree->value != -1) {
            return tree->value;
        }

        double feature_val = x[tree->feature_index % x.size()];
        if (feature_val <= tree->threshold) {
            return predict(x, tree->left);
        } else {
            return predict(x, tree->right);
        }
    }

private:

    int min_samples_split;
    int max_depth;
    Node* root;
    
    Node* build_tree(std::vector<std::vector<double>>& dataset, int depth = 0) {
        int num_samples = 0, num_features = 0;
        
        num_samples = dataset.size();
        if(num_samples)
            num_features = dataset[0].size() - 1;

        // Check stopping conditions
        if (num_samples < min_samples_split || depth >= max_depth) {
            int leaf_value = calculate_leaf_value(dataset);
            return new Node{ -1, 0.0, nullptr, nullptr, 0.0, leaf_value };
        }
        
        // Find the best split
        auto best_split = get_best_split(dataset, num_features);
        
        // Check for info gain
        if (best_split.info_gain > 0) {
            Node* left_subtree = build_tree(best_split.dataset_left, depth + 1);
            Node* right_subtree = build_tree(best_split.dataset_right, depth + 1);
            return new Node{ best_split.feature_index, best_split.threshold, left_subtree, right_subtree, best_split.info_gain, -1 };
        } else {
            // If no info gain, return a leaf node
            int leaf_value = calculate_leaf_value(dataset);
            return new Node{ -1, 0.0, nullptr, nullptr, 0.0, leaf_value };
        }
        
    }

    struct SplitResult {
        int feature_index;
        double threshold;
        double info_gain;
        std::vector<std::vector<double>> dataset_left;
        std::vector<std::vector<double>> dataset_right;
    };

    SplitResult split_dataset(const std::vector<std::vector<double>>& dataset, int feature_index, double threshold) {
        SplitResult result;

        for (const auto& row : dataset) {
            if (row[feature_index] <= threshold) {
                result.dataset_left.push_back(row);
            } else {
                result.dataset_right.push_back(row);
            }
        }

        return result;
    }

    SplitResult get_best_split(std::vector<std::vector<double>>& dataset, int num_features, std::string mode = "entropy") {
        SplitResult best_split;
        double max_info_gain = -std::numeric_limits<double>::infinity();

        for (int feature_index = 0; feature_index < num_features; ++feature_index) {
            // Sort the data along the feature to calculate possible thresholds
            std::sort(dataset.begin(), dataset.end(), 
                      [feature_index](const auto& a, const auto& b) { return a[feature_index] < b[feature_index]; });

            for (int i = 1; i < dataset.size(); ++i) {
                double threshold = (dataset[i][feature_index] + dataset[i - 1][feature_index]) / 2.0;
                auto split_data = split_dataset(dataset, feature_index, threshold);

                if (!split_data.dataset_left.empty() && !split_data.dataset_right.empty()) {
                    // Calculate information gain
                    double current_info_gain = calculate_information_gain(dataset, split_data.dataset_left, split_data.dataset_right, mode);
                    if (current_info_gain > max_info_gain) {
                        best_split = { feature_index, threshold, current_info_gain, split_data.dataset_left, split_data.dataset_right };
                        max_info_gain = current_info_gain;
                    }
                }
            }
        }

        return best_split;
    }

    double calculate_information_gain(std::vector<std::vector<double>>& y, std::vector<std::vector<double>>& y_left, std::vector<std::vector<double>>& y_right, std::string mode = "entropy") {
        double parent_loss = (mode == "gini") ? calculate_gini(y) : calculate_entropy(y);
        double child_loss = ((y_left.size() / y.size()) * ((mode == "gini") ? calculate_gini(y_left) : calculate_entropy(y_left))
                            + (y_right.size() / y.size()) * ((mode == "gini") ? calculate_gini(y_right) : calculate_entropy(y_right)));
        return parent_loss - child_loss;
    }

    double calculate_entropy(std::vector<std::vector<double>>& y) {
        std::vector<double> unique_classes;
        for (const auto& row : y) {
            unique_classes.push_back(row.back());
        }
        std::sort(unique_classes.begin(), unique_classes.end());
        unique_classes.erase(std::unique(unique_classes.begin(), unique_classes.end()), unique_classes.end());

        double entropy = 0.0;
        for (auto cls : unique_classes) {
            double p_cls = std::count_if(y.begin(), y.end(), [cls](const auto& row) { return row.back() == cls; }) / static_cast<double>(y.size());
            entropy -= p_cls * std::log2(p_cls);
        }
        return entropy;
    }

    double calculate_gini(std::vector<std::vector<double>>& y) {
        std::vector<double> unique_classes;
        for (const auto& row : y) {
            unique_classes.push_back(row.back());
        }
        std::sort(unique_classes.begin(), unique_classes.end());
        unique_classes.erase(std::unique(unique_classes.begin(), unique_classes.end()), unique_classes.end());

        double impurity = 1.0;
        for (auto cls : unique_classes) {
            double p_cls = std::count_if(y.begin(), y.end(), [cls](const auto& row) { return row.back() == cls; }) / static_cast<double>(y.size());
            impurity -= std::pow(p_cls, 2);
        }
        return impurity;
    }

    int calculate_leaf_value(std::vector<std::vector<double>>& y) {
        
        if(!y.size())   return 0;

        std::vector<int> classes;
        for (const auto& row : y) {
            classes.push_back(static_cast<int>(row.back()));
        }

        int leaf_value = *std::max_element(classes.begin(), classes.end(),
                                           [&classes](int a, int b) { return std::count(classes.begin(), classes.end(), a) < std::count(classes.begin(), classes.end(), b); });
        return leaf_value;
    }

};

// int main() {
//     // Example usage
//     std::vector<std::vector<double>> dataset, test_data;
//     std::vector<int> labels;

//     Read_Iris_Dataset(dataset, labels);

//     for(size_t i=0; i<dataset.size(); ++i){
//         test_data.push_back(dataset[i]);
//         dataset[i].push_back(labels[i]);
//     }

//     DecisionTree tree(5);
//     // std::vector<std::vector<double>> data = { {1, 2, 3, 0}, {2, 3, 4, 1}, {3, 4, 5, 0}, {4, 5, 6, 1} };
//     tree.fit(dataset);

//     std::vector<double> sample = {1.5, 3.5, 4.5};
//     int prediction = tree.predict(sample);

//     double accuracy = 0.0;
//     for(size_t i=0; i<test_data.size(); ++i){
//         if(tree.predict(test_data[i]) == labels[i]){
//             accuracy++;
//         }
//     }

//     std::cout << "Accuracy: " << accuracy/test_data.size() << std::endl;

//     return 0;
// }
