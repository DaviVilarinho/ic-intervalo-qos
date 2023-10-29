import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time

TEST_SIZE = 0.3
RANDOM_FOREST_TREES = 120

def nmae(y_pred, y_test):
    return abs(y_pred - y_test).mean() / y_test.mean()

def run_experiment(x: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series, y_metric: str, random_state=42, regression_method=None) -> dict[str, dict]:

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=random_state)
    results = {}

    if regression_method != 'random_forest':
        regression_tree_regressor = DecisionTreeRegressor() 

        training_time_reg_tree = time.time()
        regression_tree_regressor.fit(x_train, y_train)
        training_time_reg_tree = time.time() - training_time_reg_tree

        results['reg_tree'] = {
            'regressor': regression_tree_regressor,
            'nmae': nmae(regression_tree_regressor.predict(x_test), y_test[y_metric]),
            'training_time': training_time_reg_tree
        }

    if regression_method != 'reg_tree':
        random_forest_regressor = RandomForestRegressor(n_estimators=RANDOM_FOREST_TREES, random_state=random_state, n_jobs=-1)

        training_time_random_forest = time.time()
        random_forest_regressor.fit(x_train, y_train)
        training_time_random_forest = time.time() - training_time_random_forest

        results['random_forest'] = {
            'regressor': random_forest_regressor,
            'nmae': nmae(random_forest_regressor.predict(x_test), y_test[y_metric]),
            'training_time': training_time_random_forest
        }

    return results
