
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import optuna

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import xgboost as xgb
from xgboost import XGBRegressor

if __name__ == '__main__':

    csv_data = pd.read_csv('data/kink-11.csv', encoding="gbk")

    csv_x = csv_data[:, 0:column - 1]
    csv_y = csv_data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.2, random_state=6367)

    def objective(trial):
        params = {
            'objective': 'reg:squarederror', 
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),  
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), 
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),  
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),  
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True), 
        }

        model = xgb.XGBRegressor(**params)

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)

        return mse

    study = optuna.create_study(direction='minimize')

    study.optimize(objective, n_trials=300)

    print("Best hyperparameters: ", study.best_params)

    reg = XGBRegressor(**(study.best_params))
    
    reg.fit(x_train, y_train)
    
    y_pred = reg.predict(x_test)
    
    print('RMSE =', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('MAE =', mean_absolute_error(y_test, y_pred))
    print('R2 =', r2_score(y_test, y_pred))

