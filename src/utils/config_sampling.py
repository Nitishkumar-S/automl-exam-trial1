import random

def sample_config():
    """
    Randomly sample a hyperparameter configuration.
    You can later replace this with BOHB or other samplers.
    """
    return {
        "lr": 10 ** random.uniform(-4, -2),           # log-uniform between 1e-4 and 1e-2
        "dropout": random.choice([0.0, 0.2, 0.5]),     # dropout choices
        "batch_size": random.choice([32, 64, 128])     # common batch sizes
    }
