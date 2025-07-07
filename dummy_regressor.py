import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import load_data as ld


def main():
    if __name__ == '__main__':
        data = ld.read_csv("data/student-por.csv")

        X_train, y_train, X_test, y_test = ld.split_data_for_lr(data)


        dummy = DummyRegressor(strategy='mean')
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(mae, rmse, r2)

        # Write to results/stats.csv
        ld.add_stats('', 'dummy_regressor', mae, rmse, r2)

main()