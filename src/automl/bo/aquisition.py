import numpy as np

def UCB(mean, std, kappa=1.0):
    return mean - kappa * std  # Lower is better for minimization
