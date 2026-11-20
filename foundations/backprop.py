import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        n = len(x)
        z = np.dot(x,w) + b
        print(z)
        d_y = 1/(1+ np.exp(-z))  
        print(d_y) # sigmoid
        error = d_y - y_true
        d_z  = d_y*(1 - d_y)
        d_b = np.round(error * d_z, 5)
        print(d_b)
        for i in range(n):
            x[i] = np.round(x[i]*d_b,5)
        return (x, d_b)

