import numpy as np

from time import time
from random_forest import RandomForest


def accuracy_score(Y_true, Y_predict):
    Y_true = Y_true.astype(int)
    t = .0
    for i in range(np.size(Y_true)):
        if Y_true[i] == Y_predict[i]:
            t += 1
    return t / np.size(Y_predict)


# 0 - true, 1 - false
def precision(Y_true, Y_predict):
    Y_true = Y_true.astype(int)
    tp = fp = .0
    for i in range(np.size(Y_true)):
        if Y_predict[i] == 0:
            if Y_true[i] == 0:
                tp += 1
            else:
                fp += 1
    return tp / (tp + fp) if tp + fp != 0 else 0


def recall(Y_true, Y_predict):
    Y_true = Y_true.astype(int)
    tp = fn = .0
    for i in range(np.size(Y_true)):
        if Y_true[i] == 0:
            if Y_predict[i] == 0:
                tp += 1
            else:
                fn += 1
    return tp / (tp + fn) if tp + fn != 0 else 0


def evaluate_performance(trials=100):
    filename = '../data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = data[:, 0]
    n, d = X.shape

    accuracy_list = np.zeros(trials)
    precision_list = np.zeros(trials)
    recall_list = np.zeros(trials)

    for trial in range(trials):
        print 'Trial #', trial
        # Random shuffle
        idx = np.arange(n)
        np.random.seed(np.int(time() / 150))
        np.random.shuffle(idx)
        X = X[idx]
        Y = y[idx]

        # Split Train and Test samples
        train = np.int(0.8 * n)
        Xtrain = X[:train, :]
        Xtest = X[train:, :]
        Ytrain = Y[:train]
        Ytest = Y[train:]

        clf = RandomForest(n_trees=30, max_depth=100, ratio_per_tree=0.7, ratio_features=0.3)
        clf.fit(Xtrain, Ytrain)
        pred = clf.predict(Xtest)

        accuracy_list[trial] = accuracy_score(Ytest, pred)
        precision_list[trial] = precision(Ytest, pred)
        recall_list[trial] = recall(Ytest, pred)

    stats = np.zeros((3, 3))
    stats[0, 0] = np.mean(accuracy_list)
    stats[0, 1] = np.std(accuracy_list)
    stats[1, 0] = np.mean(precision_list)
    stats[1, 1] = np.std(precision_list)
    stats[2, 0] = np.mean(recall_list)
    stats[2, 1] = np.std(recall_list)
    return stats


if __name__ == '__main__':
    stats = evaluate_performance()
    print 'Accuracy= ', stats[0, 0], ' (', stats[0, 1], ')'
    print 'Precision= ', stats[1, 0], ' (', stats[1, 1], ')'
    print 'Recall= ', stats[2, 0], ' (', stats[2, 1], ')'
