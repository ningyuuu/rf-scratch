import numpy as np
from .decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=5, subsample_size=1, feature_proportion=1):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.feature_proportion = feature_proportion
        self.trees = [DecisionTree(str(i)) for i in range(self.n_trees)]

    def fit(self, X_train, y_train):
        self.feature_proportions = [self.__create_feature_samples(X_train) for i in range(self.n_trees)]
        for i in range(self.n_trees):
            tree = self.trees[i]
            X_sample, y_sample = self.__create_subsample(X_train, y_train)
            X_sample_feats = X_sample[:, self.feature_proportions[i]]
            tree.fit(X_sample_feats, y_sample)

    def predict(self, X_test):
        preds = []
        for i in range(self.n_trees):
            preds.append(self.trees[i].predict(X_test[:, self.feature_proportions[i]]))
        preds = np.stack(preds, axis=1).mean(axis=1)
        return np.rint(preds)

    def __create_feature_samples(self, X_train):
        total_feats = X_train.shape[1]
        feature_samples = np.random.choice(np.array(range(total_feats)), round(self.feature_proportion * total_feats), replace=False)
        feature_samples.sort()
        return feature_samples

    def __create_subsample(self, X_train, y_train):
        total_rows = X_train.shape[0]
        if self.subsample_size == 1:
            selector = np.random.choice(np.array(range(total_rows)), total_rows)
        else:
            selector = np.random.choice(np.array(range(total_rows)), round(self.subsample_size * total_rows), replace=False)
            
        X_sample, y_sample = X_train[selector, :], y_train[selector]
        return X_sample, y_sample

if __name__ == '__main__':
    X_train = np.array([
        [2, 3, 1, 5, 2],
        [4, 3, 1, 4, 3],
        [2, 1, 2, 2, 7],
        [2, 1, 1, 5, 2],
        [3, 3, 2, 1, 8]
    ])

    X_test = np.array([
        [3, 3, 1, 6, 2],
        [2, 3, 1, 5, 5],
        [2, 3, 2, 3, 3]
    ])

    y_train = np.array([0, 0, 1, 0, 1])

    rf = RandomForest(feature_proportion=0.6)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    print(preds)
    print(rf.feature_proportions)