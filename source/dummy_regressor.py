import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

import data_utils as du


def dr_test():
    if __name__ == '__main__':
        data_p = du.read_csv("../data/student-por.csv")
        data_m = du.read_csv("../data/student-mat.csv")

        X_train_p, y_train_p, X_test_p, y_test_p = du.split_data_for_lr(data_p)
        X_train_m, y_train_m, X_test_m, y_test_m = du.split_data_for_lr(data_m)

        for i in range(2):

            pipeline = Pipeline([
                ('regressor', DummyRegressor(strategy='mean'))
            ])

            if i == 0:
                pipeline.fit(X_train_p, y_train_p)
                y_pred = pipeline.predict(X_test_p)

                mse = mean_squared_error(y_test_p, y_pred)
                mae = mean_absolute_error(y_test_p, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_p, y_pred)

            else:
                pipeline.fit(X_train_m, y_train_m)
                y_pred = pipeline.predict(X_test_m)

                mse = mean_squared_error(y_test_m, y_pred)
                mae = mean_absolute_error(y_test_m, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_m, y_pred)

            result = [{
                'Preprocessor' : 'no_scaling',
                'Model' : 'dummy_regressor',
                'MAE': np.round(mae, 2),
                'RMSE': np.round(rmse, 2),
                'R2': np.round(r2, 2)
            }]

            # Write to results/subject/stats.csv
            if i == 0:
                du.add_stats(pd.DataFrame(result), 'por')
            else:
                du.add_stats(pd.DataFrame(result), 'mat')

dr_test()