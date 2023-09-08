from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import config_context
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

if __name__ == "__main__":
    trace = "VoD-SingleApp-PeriodicLoad" 
    
    X_trace = pd.read_csv(f'parsed/{trace}/X_parsed.csv', nrows=100)
    Y_trace = pd.read_csv(f'parsed/{trace}/Y_parsed.csv', nrows=100)
    print(X_trace.head())
    X_trace = X_trace.apply(pd.to_numeric, errors='coerce').fillna(0)
    Y_trace = Y_trace.apply(pd.to_numeric, errors='coerce').fillna(0)

    correlation = X_trace.apply(lambda x: abs(x.corr(Y_trace['DispFrames'])))
    indices = np.argsort(correlation)
    print(correlation[indices])
    """
    X_trace = X_trace[correlation.columns]

    X_train, X_test, y_train, y_test = train_test_split(X_trace, Y_trace, test_size=0.1, random_state=42)

    regression_tree = DecisionTreeRegressor() # a classification or regression decision tree is used as a predictive model to draw conclusions about a set of observations. 
    regression_tree.fit(X_train, y_train)

    joblib.dump(regression_tree, f'model_regt_{trace}.sav')

    with open(f'models/X_test_set_{trace}.csv', 'w') as FOUT:
        np.savetxt(FOUT, X_test)

    with open(f'models/Y_test_set_{trace}.csv', 'w') as FOUT:
        np.savetxt(FOUT, y_test)

    #regr_random_forest = RandomForestRegressor(n_estimators=120, random_state=0)
    #regr_random_forest.fit(X_train, y_train)

    #joblib.dump(regr_random_forest, f'models/model_rand_{trace}.sav')
    """
