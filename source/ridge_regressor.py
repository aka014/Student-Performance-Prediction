import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

import data_utils as du
import linear_regressor as lr
from features import FeatureList


def train_and_evaluate_final(X_train, y_train, X_test, y_test, subject, alpha_list, num_list, preprocessors):
    """
    Trains and evaluates the ridge regression model through different preprocessing steps.

    Parameters:
        X_train (DataFrame): Feature DataFrame of training data.
        y_train (DataFrame): Target DataFrame of training data.
        X_test (DataFrame): Feature DataFrame of test data.
        y_test (DataFrame): Target DataFrame of test data.
        subject (string): School subject of data ("por" or "mat").
        alpha_list (List): Alpha regularization parameter of Ridge regression model.
        num_list (List): Number of features to select.
        preprocessors (list): List of different preprocessors.
    """

    # Create default FeatureList
    features = FeatureList()

    # Used for stats storage
    results = []

    for name, preprocessor in preprocessors:

        alpha = alpha_list.pop(0)
        feat_num = num_list.pop(0)

        # Ridge regression estimator
        ridge = Ridge(alpha=alpha)
        # Recursive Feature Elimination setup
        rfe = RFE(estimator=ridge, n_features_to_select=feat_num)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', rfe),
            ('regressor', ridge)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Get new column names after One-Hot Encoding, etc.
        lr.update_columns(preprocessor, features)

        # Get model's coefficients
        weights_intercept = lr.create_w_b_table(pipeline, features)

        # Write coefficients to CSV file
        du.write_result(weights_intercept, f"{name}_ridge_extended_train_{alpha}_select{feat_num}", subject)

        # Calculate stats
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Append to list of results
        results.append({
            'Preprocessor': name,
            'Model' : f"ridge_extended_train_{alpha}_select{feat_num}_mae",
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
    Trains and evaluates the ridge regression model through different preprocessing steps.

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
    best_num = []

    # Used for stats storage
    results = []

    for name, preprocessor in preprocessors:

        # Combinations of these parameters will be checked if they are optimal
        param_grid = {
            'feature_selection__n_features_to_select': [20, 30, 40, 50, 60, 70],
            'regressor__alpha': [ 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 100, 250, 500, 1000]
        }

        # Ridge regression estimator
        ridge = Ridge()
        # Recursive Feature Elimination setup
        rfe = RFE(estimator=ridge)
        # Set up Cross-Validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', rfe),
            ('regressor', ridge)
        ])

        # Search for optimal parameters and set scoring method
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_absolute_error',
        )

        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_train)

        # Get best performing alpha
        alpha = grid_search.best_params_['regressor__alpha']
        # Get best performing number of features
        num = grid_search.best_params_['feature_selection__n_features_to_select']

        #print(alpha)
        #print(num)

        max_alpha.append(alpha)
        best_num.append(num)

        # Calculate stats
        r2 = r2_score(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))

        # Append to list of results
        results.append({
            'Preprocessor': name,
            'Model': f"ridge_extended_train_{alpha}_select{num}_mae_train",
            'MAE': f"{np.round(mae, 2):.2f}",
            'RMSE': f"{np.round(rmse, 2):.2f}",
            'R2': f"{np.round(r2, 2):.2f}"
        })

    # Prepare default FeatureList for next iteration (will use different preprocessing)
    features = FeatureList()

    # In the end, append every stats result to CSV file
    du.add_stats(pd.DataFrame(results), subject)

    return max_alpha, best_num

def rr_test():
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
            # ("no_scaling", ColumnTransformer([
            #     ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), features.categorical),
            #     ('num', 'passthrough', features.numerical)
            # ])),
            #
            # ("standard_scaling", ColumnTransformer([
            #     ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), features.categorical),
            #     ('num', StandardScaler(), features.numerical)
            # ])),

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

        max_alpha_p, feat_num_p = train_and_evaluate(X_train_p, y_train_p, X_test_p, y_test_p, 'por', preprocessors)
        #max_alpha_m, feat_num_m = train_and_evaluate(X_train_m, y_train_m, X_test_m, y_test_m, 'mat')


        train_and_evaluate_final(X_train_p, y_train_p, X_test_p, y_test_p, 'por', max_alpha_p, feat_num_p, preprocessors)
        #train_and_evaluate_final(X_train_m, y_train_m, X_test_m, y_test_m, 'mat', max_alpha_m, feat_num_m)


rr_test()

