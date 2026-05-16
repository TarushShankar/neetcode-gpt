import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:

        z = np.dot(x,w) + b
        y_hat = 1/(1+np.exp(-z))
        Loss = 0.5 * ((y_hat-y_true)**2)
        dL_dw = (y_hat - y_true)*y_hat*(1-y_hat)*x
        dl_dw = np.round(dL_dw,5)
        dL_db = (y_hat - y_true)*y_hat*(1-y_hat)
        dl_db = np.round(dL_db,5)
        return(dl_dw,dl_db)
        
