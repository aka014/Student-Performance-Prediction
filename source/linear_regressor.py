import numpy as np
import pandas as pd
from iso8601 import is_iso8601
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data_utils as du
from features import FeatureList


def train_and_evaluate(X_train, y_train, X_test, y_test, subject):
    """
    Trains and evaluates the linear regression model through different preprocessing steps.

    Parameters:
        X_train (DataFrame): Feature DataFrame of training data.
        y_train (DataFrame): Target DataFrame of training data.
        X_test (DataFrame): Feature DataFrame of test data.
        y_test (DataFrame): Target DataFrame of test data.
        subject (string): School subject of data ("por" or "mat").
    """

    # Create default FeatureList
    features = FeatureList()

    # Make all preprocessors
    preprocessors = [
        ("no_scaling", ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), features.categorical),
            ('num', 'passthrough', features.numerical)
        ])),

        ("standard_scaling", ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), features.categorical),
            ('num', StandardScaler(), features.numerical)
        ]))
    ]

    # Used for stats storage
    results = []

    for name, preprocessor in preprocessors:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Get new column names after One-Hot Encoding, etc.
        update_columns(preprocessor, features)

        # Get model's coefficients
        weights_intercept = create_w_b_table(pipeline, features)

        # Write coefficients to CSV file
        du.write_result(weights_intercept, f"{name}_linear_regression", subject)

        # Calculate stats
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Append to list of results
        results.append({
            'Preprocessor': name,
            'Model' : "linear_regression",
            'MAE': np.round(mae, 2),
            'RMSE': np.round(rmse, 2),
            'R2': np.round(r2, 2)
        })

        # Prepare default FeatureList for next iteration (will use different preprocessing)
        features = FeatureList()

    # In the end, append every stats result to CSV file
    du.add_stats(pd.DataFrame(results), subject)

def update_columns(preprocessor, features):
    """
    Updates column names based on preprocessor steps.

    Parameters:
        preprocessor (ColumnTransformer): Preprocessor ColumnTransformer.
        features (FeatureList): Feature list.
    """

    # Update all columns
    ohe = preprocessor.named_transformers_['cat']
    encoded_names = ohe.get_feature_names_out(features.categorical)

    # Check if polynomial features were added
    num_tr = preprocessor.named_transformers_['num']

    if isinstance(num_tr, Pipeline):
        poly = num_tr.named_steps['poly']

        if poly is not None:
            features.numerical = poly.get_feature_names_out(features.numerical).tolist()

    # Recreate all feature column headers
    all_feature_names = list(encoded_names) + features.numerical
    # Also show intercept (bias) in the CSV
    all_feature_names.append('intercept')

    features.all = all_feature_names

def create_w_b_table(pipeline, features):
    """
    Creates weights and intercept (b) DataFrame based on model's coefficients.

    Parameters:
        pipeline (Pipeline): Pipeline.
        features (FeatureList): Feature list.

    """
    # Get coefficient (weights) and intercept (bias)
    model = pipeline.named_steps['regressor']
    weights = model.coef_
    intercept = model.intercept_

    # Round values to 4 decimals
    result = np.append(weights, intercept)
    rounded_result = np.round(result, decimals=4)

    # If feature selection was performed
    if "feature_selection" in pipeline.named_steps:
        feature_mask = pipeline.named_steps['feature_selection'].support_
        features.all.remove('intercept')
        feature_names = (np.array(features.all))[feature_mask].tolist()
        feature_names.append('intercept')

        result = pd.DataFrame([rounded_result], columns=feature_names)
        return result

    result = pd.DataFrame([rounded_result], columns=features.all)

    # print(result)

    return result

def lr_test():
    if __name__ == '__main__':
        data_por = du.read_csv("../data/student-por.csv")
        data_mat = du.read_csv("../data/student-mat.csv")

        X_train_p, y_train_p, X_test_p, y_test_p = du.split_data_for_lr(data_por)
        X_train_m, y_train_m, X_test_m, y_test_m = du.split_data_for_lr(data_mat)

        train_and_evaluate(X_train_p, y_train_p, X_test_p, y_test_p, 'por')
        train_and_evaluate(X_train_m, y_train_m, X_test_m, y_test_m, 'mat')


lr_test()