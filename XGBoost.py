
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import optuna

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import xgboost as xgb
from xgboost import XGBRegressor

if __name__ == '__main__':

    csv_data = pd.read_csv('kink.csv', encoding="gbk")  

    csv_x = csv_data[:, 0:column - 1]
    csv_y = csv_data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.2, random_state=6367)

    reg = XGBRegressor(max_depth=7, learning_rate=0.10546891628382653, n_estimators=260, gamma=1.6756180208087007e-08,
                       min_child_weight=2, subsample=0.6364750762785549, colsample_bytree=0.9884882189942823,
                       alpha=1.0365361359212191e-05, reg_lambda=0.17137149507120086)

    reg.fit(x_train, y_train)

    y = reg.predict(x_train)
    
    y_pred = reg.predict(x_test)

    print('R2 =', r2_score(y_test, y_pred))
    print('RMSE =', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('MAE =', mean_absolute_error(y_test, y_pred))
    