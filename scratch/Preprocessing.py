from Algorithms import TreeInitialization
import numpy as np
from scipy.spatial.distance import pdist, squareform

class DataPreprocessing:
    def __init__(self, tree):
        self.tree = tree
        self.trees = TreeInitialization(self.tree)


    def y_to_path(self, y):
        path = np.zeros((len(y), self.tree.height+1), dtype=int)

        for i, y_i in enumerate(y):
            path_i = self.trees.nodes[y_i].path
            for j, node in enumerate(path_i):
                path[i, j] = node.id

        return path

    
    def split_data(self, X, y, train_ratio=0.7, seed=123):
        np.random.seed(seed)
        n = X.shape[0]
        path = self.y_to_path(y)

        train_id = np.random.choice(range(n), size=int(train_ratio*n), replace=False)
        while len(np.unique(path[train_id])) != self.tree.size:
            train_id = np.random.choice(range(n), size=int(train_ratio*n), replace=False)

        test_id = np.array(list(set(range(n)) - set(train_id)))

        X_train = X[train_id,]; X_test = X[test_id,]
        y_train = y[train_id]; y_test = y[test_id]

        return X_train, X_test, y_train, y_test
                

    # def feature_screening(self, X_train, y_train, num_features):
    #     p = X_train.shape[1]
    #     path = self.y_to_path(y_train)
    #     omega = np.zeros(p - 1)

    #     for s in range(1, p): 
    #         u = X_train[:, s].reshape(-1, 1)
    #         v = np.max(path, axis=1).reshape(-1, 1)
    #         omega[s - 1] = self.dcorr(u, v)
    #     select_i = np.argsort(-omega)[:num_features] + 1
    #     select_f = np.unique(np.concatenate(([0], select_i)))
        
    #     return select_f


    # def dcov(self, u, v):
    #     n = len(u)
    #     U = squareform(pdist(u, 'euclidean'))
    #     V = squareform(pdist(v, 'euclidean'))
    #     S1 = np.sum(U * V) / n**2
    #     S2 = (np.sum(U) / n**2) * (np.sum(V) / n**2)
    #     S3 = np.sum(U @ V) / n**3
    #     return np.sqrt(max(S1 + S2 - 2 * S3, 0))

    # def dcorr(self, u, v):
    #     # return self.dcov(u, v) / np.sqrt(self.dcov(u, u) * self.dcov(v, v))
    #     eps = 1e-10
    #     denom = np.sqrt(self.dcov(u, u) * self.dcov(v, v))
    #     if denom < eps:
    #         return 0
    #     return self.dcov(u, v) / denom