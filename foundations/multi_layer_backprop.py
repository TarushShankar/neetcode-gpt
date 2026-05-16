import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)

        W2 = np.array(W2)
        b2 = np.array(b2)

        y_true = np.array(y_true)


        z1 = np.dot(W1, x) + b1

        a1 = np.maximum(0, z1)

        z2 = np.dot(W2, a1) + b2

        predictions = z2

        loss = np.mean((predictions - y_true) ** 2)

        n_outputs = len(y_true)


        dpred = (2 / n_outputs) * (predictions - y_true)

        dW2 = np.outer(dpred, a1)

        db2 = dpred


        da1 = np.dot(W2.T, dpred)


        dz1 = da1 * (z1 > 0)

        dW1 = np.outer(dz1, x)

        db1 = dz1
        dW1 = np.where(np.abs(dW1) < 1e-10, 0, dW1)
        db1 = np.where(np.abs(db1) < 1e-10, 0, db1)
        dW2 = np.where(np.abs(dW2) < 1e-10, 0, dW2)
        db2 = np.where(np.abs(db2) < 1e-10, 0, db2)

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }
