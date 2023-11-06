class Node:
    def __init__(self, feature_index, threshold, left, right, info_gain, value):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value