import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        m = max(z)
        temp = []
        for i in z:
            temp.append(np.round(np.exp(i-m),4))

        s = sum(temp)
       
        n = len(temp)
        for t in range(n):
          

            temp[t] = np.round(temp[t]/s,4)
          

        return np.array(temp)
        

        
