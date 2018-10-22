import numpy as np

class DecisionTree:
    def __init__(self, leaf=''):
        self.id = leaf
        self.left = None
        self.right = None
        self.rule = (None, None)
        self.selectors = False
        self.info_gain = -float('Inf')
        self.leaf_value = None

    def fit(self, X_train, y_train):
        self.__gini_single(y_train)
        if self.gini == 1:
            self.leaf_value = y_train[0]
            return

        num_cols = X_train.shape[1]
        for col in range(num_cols):
            vals = np.unique(X_train[:, col])
            for val in vals:
                selectors = X_train[:, col] <= val
                if np.all(selectors) or np.all(~selectors):
                    continue
                new_gini = self.__gini(y_train[selectors], y_train[~selectors])
                curr_info_gain = self.gini - new_gini
                if curr_info_gain > self.info_gain:
                    self.info_gain = curr_info_gain
                    self.rule = (col, val)
                    self.selectors = selectors

        if self.selectors is False:
            self.leaf_value = round(y_train.mean())
            # print(self.leaf_value, 'is leaf value cos no selection for', self.id, X_train.shape)
            return

        self.left = DecisionTree(self.id+'l')
        self.right = DecisionTree(self.id+'r')
        # print('testing\n', X_train[self.selectors, :], '\n\n', X_train[~self.selectors, :])
        self.left.fit(X_train[self.selectors, :], y_train[self.selectors])
        self.right.fit(X_train[~self.selectors, :], y_train[~self.selectors])

    def predict(self, X_test):
        if self.leaf_value or self.leaf_value == 0:
            return self.leaf_value

        # print(self.rule, self.info_gain, self.id)
        selectors = X_test[:, self.rule[0]] <= self.rule[1]
        predictions = np.zeros_like(selectors)
        predictions[selectors] = self.left.predict(X_test[selectors])
        predictions[~selectors] = self.right.predict(X_test[~selectors])
        return predictions.astype('int')

    def __gini_single(self, y):
        self.gini = 1 - (sum(y)/len(y)) ** 2 - (1 - sum(y)/len(y)) ** 2

    def __gini(self, y1, y2):
        # print('gini debug', y1, y2)
        # https://www.researchgate.net/post/How_to_compute_impurity_using_Gini_Index
        if len(y1) == 0:
            return 1 - (sum(y2)/len(y2)) ** 2 - (1 - sum(y2)/len(y2)) ** 2
        if len(y2) == 0:  
            return 1 - (sum(y1)/len(y1)) ** 2 - (1 - sum(y1)/len(y1)) ** 2

        gini_y1 = 1 - (sum(y1)/len(y1)) ** 2 - (1 - sum(y1)/len(y1)) ** 2
        gini_y2 = 1 - (sum(y2)/len(y2)) ** 2 - (1 - sum(y2)/len(y2)) ** 2
        # print('gini values debug', gini_y1, gini_y2)
        return (len(y1) * gini_y1 + len(y2) * gini_y2) / (len(y1) + len(y2))

if __name__ == '__main__':
    def approx_eq(a, b, num_sig=5):
        return round(a, num_sig) == round(b, num_sig)

    X_train = np.array([
        [2, 3, 1],
        [4, 3, 1],
        [2, 1, 2],
        [2, 1, 1],
        [3, 3, 2]
    ])

    X_test = np.array([
        [3, 3, 1],
        [2, 3, 1]
    ])

    y_train = np.array([0, 0, 1, 0, 1])

    # print(approx_eq(DecisionTree()._DecisionTree__gini([1, 0, 0, 0, 0], [1, 1, 1, 1, 0]), .32))
    x = DecisionTree()
    x.fit(X_train, y_train)
    preds = x.predict(X_test)
    print(preds)