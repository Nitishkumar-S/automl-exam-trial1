import numpy as np
from sklearn.ensemble import RandomForestRegressor

class Surrogate:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50)
        self.X = []
        self.y = []

    def add_data(self, config_list, loss_list):
        self.X = [list(config.values()) for config in config_list]
        self.y = loss_list
        self.model.fit(self.X, self.y)

    def predict(self, configs):
        X_test = [list(config.values()) for config in configs]
        preds = self.model.predict(X_test)
        stds = np.std([tree.predict(X_test) for tree in self.model.estimators_], axis=0)
        return preds, stds
