
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import RandomSampler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':

    csv_data = pd.read_csv('kink.csv', encoding="gbk")

    csv_x = sample_data[:, 0:column-1]
    csv_y = sample_data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.2, random_state=6367)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 500) 
        max_depth = trial.suggest_int('max_depth', 1, 10) 
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

        scores = cross_val_score(model, x_train, y_train, cv=10)

        return scores.mean()

    study = optuna.create_study(direction='maximize', sampler=RandomSampler(seed=42))
    study.optimize(objective, n_trials=100) 

    reg = RandomForestRegressor(**(study.best_params))
    
    reg.fit(x_train, y_train)

    y_pred = reg.predict(x_test)
    
    print('R2 =', r2_score(y_test, y_pred))
    print('RMSE =', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('MAE =', mean_absolute_error(y_test, y_pred))

