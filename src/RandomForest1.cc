#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <random>
#include <cmath>
#include "Test.cc"
#include "iris_read.h"


// Use random_device to obtain a seed for the random number engine
std::random_device rd;

// Use the seed to initialize the random number engine
 std::mt19937_64 rng(rd());

class RandomForest {
private:
    int num_trees;
    int min_samples_split;
    int max_depth;
    int max_features;
    std::vector<DecisionTree> trees;

public:
    RandomForest(int num_trees = 100, int min_samples_split = 2, int max_depth = 7, int max_features = 0)
        : num_trees(num_trees), min_samples_split(min_samples_split), max_depth(max_depth), max_features(max_features) {}

    /// @brief Runs the logic to fit the model.
    /// @param dataset 
    /// @param labels 
    void fit(std::vector<std::vector<double>>& dataset) {
        size_t num_sub_samples = dataset.size();
        size_t num_sub_features = int(dataset[0].size()/2);

        std::cout<<"Inside of fit\n"<<std::endl;
        for (int i = 0; i < num_trees; ++i) {
            std::vector<std::vector<double>> data_subset;
            std::vector<int> label_subset;
            DecisionTree tree(min_samples_split, max_depth);
            
            // bootstrapSample(dataset, labels, num_sub_samples, num_sub_features,
            //                 data_subset, label_subset);
            std::cout<<"After bootstrap creation\n"<<std::endl;
            tree.fit(dataset);
            trees.push_back(tree);
        }
    }
    
    /// @brief Bootstrapping a sample of data
    /// @param dataset Data points which contain the parameters of a data
    /// @param labels Labels of the data points.
    /// @return Data subset vector
    void bootstrapSample(std::vector<std::vector<double>>& dataset, 
                                                        std::vector<int>& labels,
                                                        size_t num_sub_samples,
                                                        size_t num_sub_features,
                                                        std::vector<std::vector<double>>& data_subset,
                                                        std::vector<int>& label_subset) 
    {
        size_t num_samples = dataset.size();
        size_t num_features = dataset[0].size();
        std::vector<size_t> subset_indices = getRandomIndices(num_sub_samples, 0, num_samples-1, true);

        if(max_features == 0 || max_features > num_features){
            max_features = int(sqrt(num_features));
        }

        std::vector<size_t> featureIndices = getRandomIndices(max_features, 0, num_features-1, false);
        // std::vector temp = dataset[150];
        // std::cout<<"Here"<<std::endl;
        // std::cout<<"There "<<temp[0]<<std::endl;

        std::cout<<"After indices creation\n"<<std::endl;

        std::cout<<"size of dataset: "<<dataset.size()<< "size of each row: " << dataset[0].size()<<std::endl;

        std::cout<<"size of subset: "<<subset_indices.size()<< "size of feature subset: " << featureIndices.size()<<std::endl;
        
        // std::cout<<
        std::cout<<"Subset Indices:\n";
        for(auto i:subset_indices)  std::cout<<i<<" ";
        std::cout<<"\nFeature Indices:";
        for(auto i:featureIndices)  std::cout<<i<<" ";
        
        std::cout<<std::endl;

        for (int i = 0; i < subset_indices.size(); ++i) {
            std::vector<double> row;
            for (int j = 0; j < featureIndices.size(); ++j) {
                row.push_back(dataset[subset_indices[i]][featureIndices[j]]);
            }
            label_subset.push_back(labels[i]);
            data_subset.push_back(row);
        }
    }

    /// @brief Finds random numbers for a given range of numbers.
    /// @param n Number of random numbers.
    /// @param lower_bound Lower bound of the random numbers.
    /// @param upper_bound Upper bound of the random numbers.
    /// @param replace Find random numbers with replacement if True, else False.
    /// @return Vector of indices found between the lower_bound and upper_bound.
    std::vector<size_t> getRandomIndices(size_t n, size_t lower_bound, size_t upper_bound, bool replace) {

        std::uniform_int_distribution<size_t> distribution(lower_bound, upper_bound);
        std::vector<size_t> indices;

        if(replace)
            for (int i = 0; i < n; ++i) {
                indices.push_back(distribution(rng));
            }
        else{
            // Use a set to store unique random numbers
            std::set<size_t> unique_indices;
            size_t itr = 0;

            while(unique_indices.size() < n){
                size_t temp = distribution(rng);
                if(unique_indices.find(temp) == unique_indices.end()){
                    unique_indices.insert(temp);
                    indices.push_back(temp);
                }   
            }
        }
        return indices;
    }

    int predict(std::vector<double>& x) {
        std::vector<int> treePreds;
        std::cout<<"Total number of trees"<<trees.size()<<std::endl;
        for (int i = 0; i < trees.size(); ++i) {
            treePreds.push_back(trees[i].predict(x));
        }
        return getMostCommonOutput(treePreds);;
    }

    std::vector<int> predict_all(std::vector<std::vector<double>>& X) {
        std::vector<int> predictions;
        for (size_t i = 0; i < X.size(); ++i) {
            predictions.push_back(predict(X[i]));
        }
        return predictions;
    }

    int getMostCommonOutput(std::vector<int>& treePreds) {
        // Find the most voted class
        std::unordered_map<int, int> class_votes;
        int highest_voted = -1, max_votes = 0;

        for(auto a_vote:treePreds){
            if(class_votes.find(a_vote) == class_votes.end()){
                class_votes[a_vote] = 0;
            }
            class_votes[a_vote]++;
            if(class_votes[a_vote] > max_votes){
                max_votes = class_votes[a_vote];
                highest_voted = a_vote;
            }
        }

        return highest_voted;
    }

    
};

int main() {
    // Example usage of RandomForest class
    RandomForest rf(10, 3, 10);

    // Get data, parameters and labels
    
    std::vector<std::vector<double>> dataset, test_data;
    std::vector<int> labels;

    Read_Iris_Dataset(dataset, labels);

    for(size_t i=0; i<dataset.size(); ++i){
        test_data.push_back(dataset[i]);
        dataset[i].push_back(labels[i]);
    }

    std::cout<<"Read Data:";
    for(size_t i=0; i<dataset.size(); ++i){
        for(size_t j=0; j<dataset[0].size(); ++j){
            std::cout << dataset[i][j] << " ";
        }
        std::cout << labels[i] << std::endl;
    }    
    std::cout << dataset.size() << std::endl;
    // check if the data load is corr

    std::cout<<"starting with training \n"<<std::endl;
    rf.fit(dataset);

    std::cout<<"Training Ended\n"<<std::endl;
    std::vector<int> predictions = rf.predict_all(dataset);

    double accuracy = 0.0;
    for(size_t i=0; i<labels.size(); ++i){
        accuracy += ((predictions[i] == labels[i]) ? 1:0);
    }
    accuracy /= labels.size();

    std::cout<<"\n\nAccuracy is: "<<accuracy;
    // Make predictions or perform other operations with the trained forest
    // ...

    return 0;
}