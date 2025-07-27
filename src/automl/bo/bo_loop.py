from modeling.bo.surrogate import Surrogate
from modeling.bo.acquisition import UCB
from utils.config_space import get_configspace
import numpy as np

def suggest_next(configs_so_far, extrapolated_losses, stds):
    # Train surrogate
    surrogate = Surrogate()
    surrogate.add_data(configs_so_far, extrapolated_losses)

    cs = get_configspace()
    candidates = [cs.sample_configuration().get_dictionary() for _ in range(100)]

    means, uncerts = surrogate.predict(candidates)
    scores = UCB(means, uncerts, kappa=1.0)

    best_idx = np.argmin(scores)
    return candidates[best_idx]
