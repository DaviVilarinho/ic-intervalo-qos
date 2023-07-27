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
    
    X_trace = pd.read_csv(f'parsed/{trace}/X_parsed.csv')
    Y_trace = pd.read_csv(f'parsed/{trace}/Y_parsed.csv')


    X_train, X_test, y_train, y_test = train_test_split(X_trace, Y_trace, test_size=0.7, random_state=42)

    regression_tree = DecisionTreeRegressor(max_depth=2) # a classification or regression decision tree is used as a predictive model to draw conclusions about a set of observations. 
    regression_tree.fit(X_train, y_train)

    joblib.dump(regression_tree, f'model_{trace}.sav')

    with open(f'models/X_test_set_{trace}.csv', 'w') as FOUT:
        np.savetxt(FOUT, X_test)

    with open(f'models/Y_test_set_{trace}.csv', 'w') as FOUT:
        np.savetxt(FOUT, y_test)

    #regr_random_forest = RandomForestRegressor(n_estimators=120, random_state=0)
    #regr_random_forest.fit(X_train, y_train)

    #joblib.dump(regr_random_forest, f'models/model_{trace}.sav')
