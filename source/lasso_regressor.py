import numpy as np
import pandas as pd
import sklearn.linear_model
from matplotlib.widgets import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

import data_utils as du
import linear_regressor as lr
from features import FeatureList


def train_and_evaluate_final(X_train, y_train, X_test, y_test, subject, alpha_list, preprocessors):
    """
    Trains and evaluates the lasso regression model through different preprocessing steps.

    Parameters:
        X_train (DataFrame): Feature DataFrame of training data.
        y_train (DataFrame): Target DataFrame of training data.
        X_test (DataFrame): Feature DataFrame of test data.
        y_test (DataFrame): Target DataFrame of test data.
        subject (string): School subject of data ("por" or "mat").
        alpha_list (List): Alpha regularization parameter of Ridge regression model.
        preprocessors (list): List of different preprocessors.
    """

    # Create default FeatureList
    features = FeatureList()

    # Used for stats storage
    results = []

    for name, preprocessor in preprocessors:

        alpha = alpha_list.pop(0)

        # Lasso regression estimator
        lasso = Lasso(max_iter=100000, tol=1e-5, alpha=alpha)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', lasso)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Get new column names after One-Hot Encoding, etc.
        lr.update_columns(preprocessor, features)

        # Get model's coefficients
        weights_intercept = lr.create_w_b_table(pipeline, features)

        # Write coefficients to CSV file
        du.write_result(weights_intercept, f"{name}_lasso_extended_train_{alpha:.4f}", subject)

        # Calculate stats
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Append to list of results
        results.append({
            'Preprocessor': name,
            'Model' : f"lasso_extended_train_{alpha:.4f}_mae",
            'MAE': f"{np.round(mae, 2):.2f}",
            'RMSE': f"{np.round(rmse, 2):.2f}",
            'R2': f"{np.round(r2, 2):.2f}"
        })

        # Prepare default FeatureList for next iteration (will use different preprocessing)
        features = FeatureList()

    # In the end, append every stats result to CSV file
    du.add_stats(pd.DataFrame(results), subject)


def train_and_evaluate(X_train, y_train, X_test, y_test, subject, preprocessors):
    """
    Trains and evaluates the lasso regression model through different preprocessing steps.

    Parameters:
        X_train (DataFrame): Feature DataFrame of training data.
        y_train (DataFrame): Target DataFrame of training data.
        X_test (DataFrame): Feature DataFrame of test data.
        y_test (DataFrame): Target DataFrame of test data.
        subject (string): School subject of data ("por" or "mat").
        preprocessors (list): List of different preprocessors.

    Returns:
        max_alpha (List): Best performing regularization value for each preprocessor.
    """

    # Create default FeatureList
    features = FeatureList()

    # To keep best results
    max_alpha = []

    # Used for stats storage
    results = []

    for name, preprocessor in preprocessors:

        # Combinations of these parameters will be checked if they are optimal
        param_grid = {
            'regressor__alpha': np.logspace(-3, 0, num=100)
            #'regressor__alpha': np.round(np.arange(0.001, 1.001, 0.001), 3)
        }

        # Lasso regression estimator
        lasso = Lasso(max_iter=10000, tol=1e-4)
        # Set up Cross-Validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', lasso)
        ])

        # Search for optimal parameters and set scoring method
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_absolute_error',
        )

        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)

        # Get best performing alpha
        alpha = grid_search.best_params_['regressor__alpha']

        print("Iterations used:", grid_search.best_estimator_.named_steps['regressor'].n_iter_)

        #print(alpha)
        #print(num)

        max_alpha.append(alpha)

        # Calculate stats
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Append to list of results
        results.append({
            'Preprocessor': name,
            'Model': f"lasso_train_{alpha:.4f}_mae",
            'MAE': f"{np.round(mae, 2):.2f}",
            'RMSE': f"{np.round(rmse, 2):.2f}",
            'R2': f"{np.round(r2, 2):.2f}"
        })

    # Prepare default FeatureList for next iteration (will use different preprocessing)
    features = FeatureList()

    # In the end, append every stats result to CSV file
    du.add_stats(pd.DataFrame(results), subject)

    return max_alpha

def lasso_test():
    if __name__ == '__main__':

        data_por = du.read_csv("../data/student-por.csv")
        #data_mat = du.read_csv("../data/student-mat.csv")

        features = FeatureList()

        num_pipeline1 = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=True)),
        ])

        num_pipeline2 = Pipeline(steps=[
            ('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=True)),
            ('scaler', StandardScaler())
        ])

        # Make all preprocessors
        preprocessors = [
            ("no_scaling", ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), features.categorical),
                ('num', 'passthrough', features.numerical)
            ])),

            ("standard_scaling", ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), features.categorical),
                ('num', StandardScaler(), features.numerical)
            ])),

            ("standard_scaling_poly", ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), features.categorical),
                ('num', num_pipeline1, features.numerical)
            ])),

            ("poly_standard_scaling", ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), features.categorical),
                ('num', num_pipeline2, features.numerical)
            ]))
        ]


        X_train_p, y_train_p, X_test_p, y_test_p = du.split_data_for_lr(data_por)
        #X_train_m, y_train_m, X_test_m, y_test_m = du.split_data_for_lr(data_mat)

        max_alpha_p = train_and_evaluate(X_train_p, y_train_p, X_train_p, y_train_p, 'por', preprocessors)
        # max_alpha_m = train_and_evaluate(X_train_m, y_train_m, X_train_m, y_train_m, 'mat', preprocessors)


        train_and_evaluate_final(X_train_p, y_train_p, X_test_p, y_test_p, 'por', max_alpha_p, preprocessors)
        #train_and_evaluate_final(X_train_m, y_train_m, X_test_m, y_test_m, 'mat', max_alpha_m, preprocessors)


lasso_test()

# notice the difference in convergence time between scale + poly and poly +scale
# first takes 47 iterations, while second takes 438 iterations

