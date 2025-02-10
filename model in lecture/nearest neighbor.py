import numpy as np

class NearestNeighbor:
    """A terrible 'algorithm',train takes O(1) but inference takes O(N),use as an example"""
    def __init__(self):
        pass
    def train(self,X,Y):
        """X is N x D where each row is an example, y is 1-dimension of size N"""
        # The nearest neighbor just simply remembers all the training data
        self.X_train = X
        self.Y_train = Y

    def predict(self,X):
        """X is N x D"""
        num_test = X.shape[0]
        Y_pred = np.zeros(num_test, dtype = self.Y_train.dtype)
        for i in range(num_test):
            distances = np.sum(np.abs(self.X_train - X[i, :]), axis = 1)
            min_index = np.argmin(distances)
            Y_pred[i] = self.Y_train[min_index]

        return Y_pred