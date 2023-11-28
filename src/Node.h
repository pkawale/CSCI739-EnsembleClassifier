struct Node {
    int feature_index;
    double threshold;
    Node* left;
    Node* right;
    double info_gain;
    int value;
};