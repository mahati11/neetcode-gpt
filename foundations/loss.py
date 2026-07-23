import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        res = 0
        n = len(y_true)
        for i in range(n):
            if y_true[i] == 1:
                print(y_true[i])
                res += np.log(y_pred[i] + (1e-7))
            else:
                res += np.log((1-y_pred[i]) + (1e-7))
        
        ans = -res/n

        return np.round(ans,4)



    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        res= 0
        n = len(y_true)
        m = len(y_true[0])
        for i in range(n):
            for j in range(m):
                if y_true[i][j] == 1:
                    res+= np.log(y_pred[i][j] + 1e-7)
        ans = -res/n
        return np.round(ans,4)

