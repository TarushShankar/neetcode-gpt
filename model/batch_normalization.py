import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)

        if training:
            # Step 1: compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            # Step 2: normalize
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)
            # Step 3: update running stats
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var = (1 - momentum) * running_var + momentum * batch_var
        else:
            # Inference: use running stats
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        # Affine transform
        out = gamma * x_hat + beta

        # Round and convert back to list
        return (np.round(out, 4).tolist(),
                np.round(running_mean, 4).tolist(),
                np.round(running_var, 4).tolist())
