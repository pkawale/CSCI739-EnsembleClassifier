#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <cmath>
#include <random>
#include <cmath>
#include <getopt.h>
#include "DecisionTree.cc"
#include "iris_read.h"
#include "InputRead.h"


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
    void fit(std::vector<std::vector<double>>& dataset, std::string mode="entropy") {

        // std::cout<<"Inside of fit\n"<<std::endl;
        #pragma omp parallel for
        for (int i = 0; i < num_trees; ++i) {
            std::vector<std::vector<double>> data_subset;
            std::vector<int> label_subset;
            DecisionTree tree(min_samples_split, max_depth);
            
            bootstrapSample(dataset,
                            data_subset);
            // std::cout<<"After bootstrap creation\n"<<std::endl;
            tree.fit(data_subset, mode);
            trees.push_back(tree);
        }
    }
    
    /// @brief Bootstrapping a sample of data
    /// @param dataset Data points which contain the parameters of a data
    /// @param labels Labels of the data points.
    /// @return Data subset vector
    void bootstrapSample(std::vector<std::vector<double>>& dataset,
                            std::vector<std::vector<double>>& data_subset) 
    {
        size_t n_samples = dataset.size();
        size_t n_features = dataset[0].size() - 1;
        std::vector<size_t> subset_indices = getRandomIndices(n_samples, 0, n_samples-1, true);

        if(max_features == 0 || max_features > n_features){
            max_features = int(sqrt(n_features));
        }

        std::vector<size_t> featureIndices = getRandomIndices(max_features, 0, n_features-1, false);

        for (int i = 0; i < subset_indices.size(); ++i) {
            std::vector<double> row;
            for (int j = 0; j < featureIndices.size(); ++j) {
                row.push_back(dataset[subset_indices[i]][featureIndices[j]]);
            }
            row.push_back(dataset[subset_indices[i]][n_features]);
            // label_subset.push_back(labels[i]);
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

int main(int argc, char* argv[]) {
    
    // Initialize default values
    int num_trees = 100;
    int min_samples_split = 3;
    int max_depth = 10;
    int max_features = 0;
    std::string ip_type = "iris", tree_type = "entropy";

    // Parse command-line arguments
    int opt;
    while ((opt = getopt(argc, argv, "n:s:d:f:t:i:")) != -1) {
        switch (opt) {
            case 'n':
                {
                    char *endptr;
                    float temp = strtof(optarg, &endptr);

                    if (*endptr != '\0' && *endptr != '\n') {
                        std::cerr << "Invalid input for -n: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    if (std::trunc(temp) != temp) {
                        std::cerr << "Non-integer input for -n: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    num_trees = static_cast<int>(temp);
                    if(num_trees < 0){
                        std::cerr << "Negative-integer input for -n: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                break;
            case 's':
                {
                    char *endptr;
                    float temp = strtof(optarg, &endptr);

                    if (*endptr != '\0' && *endptr != '\n') {
                        std::cerr << "Invalid input for -s: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    if (std::trunc(temp) != temp) {
                        std::cerr << "Non-integer input for -s: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    min_samples_split = static_cast<int>(temp);
                    if(min_samples_split < 0){
                        std::cerr << "Negative-integer input for -s: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                break;
            case 'd':
                {
                    char *endptr;
                    float temp = strtof(optarg, &endptr);

                    if (*endptr != '\0' && *endptr != '\n') {
                        std::cerr << "Invalid input for -d: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    if (std::trunc(temp) != temp) {
                        std::cerr << "Non-integer input for -d: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    max_depth = static_cast<int>(temp);
                    if(max_depth < 0){
                        std::cerr << "Negative-integer input for -d: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                break;
            case 'f':
                {
                    char *endptr;
                    float temp = strtof(optarg, &endptr);

                    if (*endptr != '\0' && *endptr != '\n') {
                        std::cerr << "Invalid input for -f: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    if (std::trunc(temp) != temp) {
                        std::cerr << "Non-integer input for -f: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    max_features = temp;
                    if(max_features < 0){
                        std::cerr << "Negative-integer input for -f: " << optarg << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                break;
            case 't':
                tree_type = optarg;
                if(strcmp(tree_type.c_str(), "gini") && strcmp(tree_type.c_str(), "entropy")){
                    std::cerr << "Incorrect Tree type: " << optarg << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'i':
                ip_type = optarg;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-n <num_trees>]\n"
                                                     "[-s <min_samples_split>]\n"
                                                     "[-d <max_depth>]\n"
                                                     "[-f <max_features>]\n" 
                                                     "[-i <input-type: iris/mnist>]\n"
                                                     "[-t <tree-type: gini/entropy>]\n"
                                                     "[-h this help message]" << std::endl;
                exit(EXIT_FAILURE);
        }
    }
    

    RandomForest rf(num_trees, min_samples_split, max_depth);

    // Get data, parameters and labels
    
    std::vector<std::vector<double>> dataset, test_data;
    std::vector<int> labels;
    
    std::cout<<"Loading data..."<<std::endl;

    if(strcmp(ip_type.c_str(), "mnist") == 0){
        // Read training data
        read_mnist("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", dataset);
        // Read testing data
        read_mnist("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", test_data);

        for(size_t i=0; i<test_data.size(); ++i){
            std::vector<double> temp;
            labels.push_back(test_data[i][test_data[i].size() - 1]);
            for(size_t j=0; j<test_data[i].size()-1; ++j)
                temp.push_back(test_data[i][j]);
            test_data[i] = temp;
        }
    }

    else if(strcmp(ip_type.c_str(), "iris") == 0){
        Read_Iris_Dataset(dataset, labels);
        
        for(size_t i=0; i<dataset.size(); ++i){
            test_data.push_back(dataset[i]);
            dataset[i].push_back(labels[i]);
        }
    }

    else{
        std::cerr << "Unknown input for -t: " +ip_type<< std::endl;
        exit(EXIT_FAILURE);
    }

    
    if(test_data.size()){
        if(max_features > test_data[0].size()){
            std::cerr << "Max features is greater than the total features in data."<< std::endl;
            exit(EXIT_FAILURE);
        }
    }

    if(min_samples_split > dataset.size()){
        std::cerr << "Min samples to split is greater than the total data."<< std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout<<"Starting with training..."<<std::endl;
    rf.fit(dataset);

    std::cout<<"Training Ended\n";
    std::vector<int> predictions = rf.predict_all(dataset);

    double accuracy = 0.0;
    for(size_t i=0; i<labels.size(); ++i){
        accuracy += ((predictions[i] == labels[i]) ? 1:0);
    }
    accuracy /= labels.size();

    std::cout<<"\n\nAccuracy is: "<<accuracy*100.0<<"%"<<std::endl;
    
    return 0;
}