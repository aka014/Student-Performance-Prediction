import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

import data_utils as du
import linear_regressor as lr
from features import FeatureList


def train_and_evaluate_final(X_train, y_train, X_test, y_test, subject, alpha_list, l1_list, preprocessors):
    """
    Trains and evaluates the elastic net model through different preprocessing steps.

    Parameters:
        X_train (DataFrame): Feature DataFrame of training data.
        y_train (DataFrame): Target DataFrame of training data.
        X_test (DataFrame): Feature DataFrame of test data.
        y_test (DataFrame): Target DataFrame of test data.
        subject (string): School subject of data ("por" or "mat").
        alpha_list (List): Alpha regularization parameter of Ridge regression model.
        l1_list (List): L1 parameter of Elastic Net regression model.
        preprocessors (list): List of different preprocessors.
    """

    # Create default FeatureList
    features = FeatureList()

    # Used for stats storage
    results = []

    for name, preprocessor in preprocessors:

        alpha = alpha_list.pop(0)
        best_l1 = l1_list.pop(0)

        # Elastic Net regression estimator
        e_net = ElasticNet(alpha=alpha, max_iter = 100000, tol = 1e-5, l1_ratio = best_l1)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', e_net)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Get new column names after One-Hot Encoding, etc.
        lr.update_columns(preprocessor, features)

        # Get model's coefficients
        weights_intercept = lr.create_w_b_table(pipeline, features)

        # Write coefficients to CSV file
        du.write_result(weights_intercept, f"{name}_elastic_net_extended_train_{alpha:.4f}", subject)

        # Calculate stats
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Append to list of results
        results.append({
            'Preprocessor': name,
            'Model' : f"elastic_net_extended_train_{alpha:.4f}_mae",
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
    Trains and evaluates the Elastic Net regression model through different preprocessing steps.

    Parameters:
        X_train (DataFrame): Feature DataFrame of training data.
        y_train (DataFrame): Target DataFrame of training data.
        X_test (DataFrame): Feature DataFrame of test data.
        y_test (DataFrame): Target DataFrame of test data.
        subject (string): School subject of data ("por" or "mat").
        preprocessors (list): List of different preprocessors.

    Returns:
        max_alpha (List): Best performing regularization value for each preprocessor.
        l1_list (List): L1 parameter of Elastic Net regression model.
    """

    # Create default FeatureList
    features = FeatureList()

    # To keep best results
    max_alpha = []
    l1_list = []

    # Used for stats storage
    results = []

    for name, preprocessor in preprocessors:

        # Combinations of these parameters will be checked if they are optimal
        param_grid = {
            'regressor__alpha': np.logspace(-3, 0, num=100),
            #'regressor__alpha': np.round(np.arange(0.001, 1.001, 0.001), 3)
            'regressor__l1_ratio': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        }

        # Elastic Net regression estimator
        e_net = ElasticNet(max_iter=100000, tol = 1e-4)
        # Set up Cross-Validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', e_net)
        ])

        # Search for optimal parameters and set scoring method
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)

        # Get best performing alpha
        alpha = grid_search.best_params_['regressor__alpha']
        best_l1 = grid_search.best_params_['regressor__l1_ratio']

        print("Iterations used:", grid_search.best_estimator_.named_steps['regressor'].n_iter_)

        # print(alpha)
        # print(best_l1)

        max_alpha.append(alpha)
        l1_list.append(best_l1)

        # Calculate stats
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Append to list of results
        results.append({
            'Preprocessor': name,
            'Model': f"elastic_net_train_{alpha:.4f}_mae",
            'MAE': f"{np.round(mae, 2):.2f}",
            'RMSE': f"{np.round(rmse, 2):.2f}",
            'R2': f"{np.round(r2, 2):.2f}"
        })

    # Prepare default FeatureList for next iteration (will use different preprocessing)
    features = FeatureList()

    # In the end, append every stats result to CSV file
    du.add_stats(pd.DataFrame(results), subject)

    return max_alpha, l1_list

def elastic_net_test():
    data_por = du.read_csv("../data/student-por.csv")
    data_mat = du.read_csv("../data/student-mat.csv")

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
    X_train_m, y_train_m, X_test_m, y_test_m = du.split_data_for_lr(data_mat)

    # Get the best parameters using CV
    max_alpha_p, best_l1_p = train_and_evaluate(X_train_p, y_train_p, X_train_p, y_train_p, 'por', preprocessors)
    max_alpha_m, best_l1_m = train_and_evaluate(X_train_m, y_train_m, X_train_m, y_train_m, 'mat', preprocessors)

    # Final test using the best parameters
    train_and_evaluate_final(X_train_p, y_train_p, X_test_p, y_test_p, 'por', max_alpha_p, best_l1_p, preprocessors)
    train_and_evaluate_final(X_train_m, y_train_m, X_test_m, y_test_m, 'mat', max_alpha_m, best_l1_m, preprocessors)


if __name__ == '__main__':
    elastic_net_test()

# opet velika razlika izmedju scale + poly i poly + scale, prvi 17 iteracija, drugi 143
# spomeni kakva cuda cini n_jobs