import numpy as np

from time import time


class DecisionNode(object):
    def __init__(self,
                 feature=None,
                 value=None,
                 false_branch=None,
                 true_branch=None,
                 is_leaf=False,
                 current_results=None
                 ):
        self.feature = feature
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.is_leaf = is_leaf
        self.current_results = current_results


def label_frequencies(data):
    n, d = data.shape
    labels = data[:, (d - 1):].reshape(n).astype(int)
    return np.bincount(labels)


def different_labels(labels):
    c = 0
    for label in labels:
        if label != 0:
            c += 1
    return c


def split_data(data, feature, value):
    t = []
    f = []
    for row in data:
        if row[feature] >= value:
            t.append(row)
        else:
            f.append(row)
    t = np.array(t)
    f = np.array(f)
    return t, f


def entropy(data):
    n = len(data)
    if n == 0:
        return 0
    labels = label_frequencies(data).astype(float) / n
    e = .0
    for label in labels:
        if label > .0:
            e -= label * np.log2(label)
    return e


def entropy_for_feature(data, feature):
    values = np.unique(data[:, feature])
    entropy_loss = 1e5
    value = 0
    for val in values:
        t, f = split_data(data, feature, val)
        curr_entropy_loss = (len(t) * entropy(t) + len(f) * entropy(f)) / len(data)
        if entropy_loss > curr_entropy_loss:
            entropy_loss = curr_entropy_loss
            value = val
    return entropy_loss, value


def build_tree(data, depth=0, ratio_features=1.0, max_depth=100):
    n, d = data.shape
    if n == 0:
        return DecisionNode(is_leaf=True)

    current_results = label_frequencies(data)
    if depth == max_depth:
        return DecisionNode(current_results=current_results, is_leaf=True)
    if different_labels(current_results) == 1:
        return DecisionNode(current_results=current_results, is_leaf=True)

    # Select Random Features
    seed = np.int(time() / 150)
    idx = np.arange(d - 1)
    np.random.seed(seed)
    np.random.shuffle(idx)
    idx = idx[: np.int(ratio_features * (d - 1))]

    entropy_loss = 1e5
    feature = None
    value = None
    for feat in idx:
        entropy, val = entropy_for_feature(data, feat)
        if entropy_loss > entropy:
            entropy_loss = entropy
            feature = feat
            value = val

    t, f = split_data(data, feature, value)
    return DecisionNode(
        feature=feature,
        value=value,
        false_branch=build_tree(f, depth + 1, ratio_features, max_depth),
        true_branch=build_tree(t, depth + 1, ratio_features, max_depth),
        current_results=current_results
    )


def pred(tree, x):
    if tree.is_leaf or different_labels(tree.current_results) == 1:
        return np.argmax(tree.current_results)
    else:
        if x[tree.feature] >= tree.value:
            return pred(tree.true_branch, x)
        else:
            return pred(tree.false_branch, x)


class RandomTree(object):
    def __init__(self, max_depth=10, ratio_features=1.0):
        self.tree = None
        self.max_depth = max_depth
        self.ratio_features = ratio_features

    def fit(self, X, Y):
        Y = Y.reshape((1, X.shape[0])).T
        data = np.concatenate((X, Y), axis=1)
        self.tree = build_tree(data, ratio_features=self.ratio_features, max_depth=self.max_depth)

    def predict(self, X):
        return np.array([pred(self.tree, x) for x in X])
