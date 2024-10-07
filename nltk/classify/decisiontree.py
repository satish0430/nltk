# nltk/classify/decision_tree.py

from nltk.classify import ClassifierI
from nltk.probability import FreqDist
import math


class DecisionTreeClassifier(ClassifierI):
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def train(self, labeled_featuresets):
        features = list(labeled_featuresets[0][0].keys())
        self.tree = self._build_tree(labeled_featuresets, features, 0)

    def _build_tree(self, data, features, depth):
        labels = [label for _, label in data]
        if len(set(labels)) == 1:
            return labels[0]
        if depth >= self.max_depth or not features:
            return max(set(labels), key=labels.count)

        best_feature = max(features, key=lambda f: self._information_gain(data, f))
        tree = {best_feature: {}}

        for value in set(sample[0][best_feature] for sample in data):
            subset = [sample for sample in data if sample[0][best_feature] == value]
            new_features = [f for f in features if f != best_feature]
            tree[best_feature][value] = self._build_tree(subset, new_features, depth + 1)

        return tree

    def _information_gain(self, data, feature):
        def entropy(labels):
            freqdist = FreqDist(labels)
            probs = [count / len(labels) for count in freqdist.values()]
            return -sum(p * math.log2(p) for p in probs)

        labels = [label for _, label in data]
        feature_values = set(sample[0][feature] for sample in data)

        base_entropy = entropy(labels)
        weighted_feature_entropy = sum(
            len([sample for sample in data if sample[0][feature] == value]) / len(data) *
            entropy([sample[1] for sample in data if sample[0][feature] == value])
            for value in feature_values
        )

        return base_entropy - weighted_feature_entropy

    def classify(self, featureset):
        def _classify(tree, featureset):
            if not isinstance(tree, dict):
                return tree
            feature = next(iter(tree))
            if featureset[feature] in tree[feature]:
                return _classify(tree[feature][featureset[feature]], featureset)
            return max(tree[feature].values(), key=lambda x: x if not isinstance(x, dict) else 0)

        return _classify(self.tree, featureset)

    def show_most_informative_features(self, n=10):
        # This method is left as an exercise. It should return the top n most informative features.
        pass

    class DecisionTreeClassifier(ClassifierI):
        """
        A decision tree classifier for text classification tasks.

        This classifier builds a decision tree based on the information gain
        of features in the training data.

        Parameters:
        -----------
        max_depth : int, optional (default=5)
            The maximum depth of the decision tree.
        """

        def train(self, labeled_featuresets):
            """
            Train the decision tree classifier.

            Parameters:
            -----------
            labeled_featuresets : list of (dict, label) tuples
                The training data, where each tuple contains a feature dictionary
                and its corresponding label.
            """
            # ... (implementation)

        # Add similar docstrings for other methods