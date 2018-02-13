import numpy as np
from collections import Counter

def findMode(options):
    c = Counter(options)
    return c.most_common(1)[0][0]

def calc_entropy(p):
    if p != 0:
        return -p * np.log2(p)
    else:
        return 0


class Id3Model:
    tree = {}
    featureNames = []
    rootFeature = ""

    def __init__(self, data, targets, featureNames):
        self.featureNames = featureNames
        self.tree = self.make_tree(data, targets, featureNames)

    # recursively navigates the tree
    def searchTree(self, tree, row, featureNames):

        if type(tree) is dict:
            for feature in featureNames:
                if feature in tree:
                    if row[feature] in tree[feature]:
                        subtree = tree[feature][row[feature]]
                    else:
                        # in case we didn't see this option in the train data
                        subtree = next (iter (tree[feature].values()))
                    return self.searchTree(subtree, row, featureNames)
        else:
            # not a dictionary. Must be a leaf
            return tree


    def predict(self, data):
        result = []
        for index, row in data.iterrows():
            result.append(self.searchTree(self.tree, row, self.featureNames))
        return result

    def calc_info_gain(self, data, targets, feature):
        gain = 0
        options = data[feature].unique()

        for index in range(len(options)):
            option = options[index]
            entropy = 0
            newTargets = targets[data[feature] == option]
            classCounts = newTargets.value_counts()

            for classCount in classCounts:
                entropy -= calc_entropy(float(classCount)) / len(newTargets)

            gain += float(len(data.columns)) / data.shape[0] * entropy
        return gain

    def make_tree(self, data, targets, featureNames):
        tree = {}
        # all examples have the same label
        if len(set(targets)) == 1:
            return targets.iloc[0]

        # no features left to test or have already gone deep enough
        if len(data.columns) == 0:
            return max(set(targets), key=targets.count)

        # find the best feature
        bestInfoGain = 0
        bestFeature = featureNames[0]
        for feature in featureNames:
            gain = self.calc_info_gain(data, targets, feature)
            if gain > bestInfoGain:
                bestInfoGain = gain
                bestFeature = feature

        # make a node for the best feature
        tree[bestFeature] = {}

        options = set(data[bestFeature])
        for option in options:
            newData = data[data[bestFeature] == option]
            newTargets = targets[data[bestFeature] == option]
            newData = newData.drop(bestFeature, axis=1)
            featureNames = [x for x in featureNames if x != bestFeature]
            tree[bestFeature][option] = self.make_tree(newData, newTargets, featureNames)
        return tree

class Id3Classifier:

    def fit(self, data, target, featureNames):
        return Id3Model(data, target, featureNames)