import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        t = np.dot(X, weights)
        return np.round(t, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        n =len(ground_truth)
        s = 0
        for i in range(n):
            s+= (ground_truth[i][0] - model_prediction[i][0])**2
        return np.round(s/n, 5)

