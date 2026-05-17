import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        

        a = x

        for i in range(len(weights)):

            z = a @ weights[i] + biases[i]

            if i != len(weights) - 1:
                a = np.maximum(0,z)
            else:
                a = z
            

            
        return np.round(a,5)
