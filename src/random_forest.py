import numpy as np

from time import time
from random_tree import RandomTree


class RandomForest(object):

    def __init__(self, n_trees=10, max_depth=10, ratio_per_tree=0.5, ratio_features=1.0):
        self.trees = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.ratio_per_tree = ratio_per_tree
        self.ratio_features = ratio_features

    def fit(self, X, Y):
        n, d = X.shape
        self.trees = []

        for i in range(self.n_trees):
            idx = np.arange(n)
            np.random.seed(np.int(time() / 150))
            np.random.shuffle(idx)
            X = X[idx]
            Y = Y[idx]

            train = np.int(self.ratio_per_tree * n)
            Xtrain = X[:train, :]
            Ytrain = Y[:train]

            clf = RandomTree(max_depth=self.max_depth, ratio_features=self.ratio_features)
            clf.fit(Xtrain, Ytrain)
            self.trees.append(clf)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees]).T
        predictions = np.array([np.argmax(np.bincount(prediction)) for prediction in predictions])
        return predictions
