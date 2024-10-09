
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import optuna

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':

    csv_data = pd.read_csv('kink.csv', encoding="gbk")

    csv_x = sample_data[:, 0:column-1]
    csv_y = sample_data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.2, random_state=6367)

    reg = RandomForestRegressor(n_estimators=26, max_depth=10, min_samples_split=3, min_samples_leaf=1, random_state=42)
    
    reg.fit(x_train, y_train)

    y_pred = reg.predict(x_test)
    
    print('R2 =', r2_score(y_test, y_pred))
    print('RMSE =', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('MAE =', mean_absolute_error(y_test, y_pred))

