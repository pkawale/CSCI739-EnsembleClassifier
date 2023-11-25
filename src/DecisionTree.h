#include "Node.h"

class DecisionTree {
private:
    int min_samples_split;
    int max_depth;
    Node* root;

public:
    DecisionTree(int min_samples_split, int max_depth = 7)
        : min_samples_split(min_samples_split), max_depth(max_depth), root(nullptr) {}

    void fit(std::vector<std::vector<double>>& X, std::vector<double> y) {};

    Node* buildTree(std::vector<std::vector<double>>& X, std::vector<double>& y, size_t N, size_t D, int depth = 0) {};

    int calculateLeafValue(std::vector<double>& y) {};


};