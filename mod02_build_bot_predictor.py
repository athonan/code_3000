# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=100,
        subsample=0.0001,
        min_samples_leaf=0.0001,
        random_state=seed
    )
    model.fit(X, y)
    return model