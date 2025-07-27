from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter

def get_configspace():
    cs = ConfigurationSpace()

    cs.add_hyperparameter(UniformFloatHyperparameter("lr", lower=1e-4, upper=1e-2, log=True))
    cs.add_hyperparameter(CategoricalHyperparameter("dropout", choices=[0.0, 0.2, 0.5]))
    cs.add_hyperparameter(UniformIntegerHyperparameter("batch_size", lower=32, upper=128, q=32))

    return cs
