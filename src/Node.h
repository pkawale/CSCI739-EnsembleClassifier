
class Node {
public:
    int feature_index;
    double threshold;
    Node* left;
    Node* right;
    double info_gain;
    int value;

    // Constructor for node
    Node(int feature_index, double threshold, Node* left, Node* right, double info_gain, int value)
        : feature_index(feature_index), 
        threshold(threshold), 
        left(left), 
        right(right), 
        info_gain(info_gain), 
        value(value) {}
};