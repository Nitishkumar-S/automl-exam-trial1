# class LearningCurveExtrapolator:
#     def __init__(self):
#         self.history = []

#     def fit(self, epochs, train_scores, val_scores):
#         self.history.append((epochs, train_scores, val_scores))

#     def predict(self, future_epochs):
#         if not self.history:
#             raise ValueError("No data to extrapolate from.")
        
#         # Simple linear extrapolation based on the last recorded scores
#         last_epochs, last_train_scores, last_val_scores = self.history[-1]
#         if future_epochs <= last_epochs:
#             raise ValueError("Future epochs must be greater than the last recorded epochs.")
        
#         # Calculate the slope for training and validation scores
#         train_slope = (last_train_scores[-1] - last_train_scores[0]) / last_epochs
#         val_slope = (last_val_scores[-1] - last_val_scores[0]) / last_epochs
        
#         # Extrapolate future scores
#         future_train_score = last_train_scores[-1] + train_slope * (future_epochs - last_epochs)
#         future_val_score = last_val_scores[-1] + val_slope * (future_epochs - last_epochs)
        
#         return future_train_score, future_val_score

import numpy as np
from scipy.optimize import curve_fit

def exp_func(t, a, b, c):
    return a * np.exp(-b * t) + c

def extrapolate(config, val_losses, observed_epochs, target_epoch):
    """
    Args:
        config: dict, hyperparameter config (not used yet)
        val_losses: list of floats, validation losses observed so far
        observed_epochs: list of ints, epochs at which val_losses were measured
        target_epoch: int, where we want to predict the loss

    Returns:
        (predicted_loss_at_target_epoch, estimated_std)
    """
    x = np.array(observed_epochs)
    y = np.array(val_losses)

    # Initial guess for parameters a, b, c
    try:
        popt, _ = curve_fit(exp_func, x, y, p0=(1.0, 0.1, 0.1), maxfev=10000)
        pred = exp_func(target_epoch, *popt)
    except Exception as e:
        # Fallback to last known loss if fitting fails
        pred = y[-1]

    # Baseline: fixed std estimate
    return pred, 0.05
