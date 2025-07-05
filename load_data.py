import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def read_csv(file_path, drop_grades=True):
    """"
    Reads a CSV file and returns a pandas dataframe.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        DataFrame: Dataset.
    """

    # Read CSV file
    df = pd.read_csv(file_path, sep=';')

    # These features are obviously the most important ones, so the model will be tested without them
    #maybe make drop grades a class attribute in order to get it for writing!!
    if drop_grades:
       df.drop(columns=['G1', 'G2'], inplace=True)

    return df


def split_data_for_lr(df, train='GP'):
    """
    Splits dataset into training and test sets.

    Parameters:
        df (DataFrame): Dataset.
        train (string): 'GP' or 'MS'

    Returns:
        tuple: X_train, y_train, X_test, y_test
        X - input features
        y - target values
    """

    if train == 'GP': test = 'MS'
    elif train == 'MS': test = 'GP'
    else:
        train = 'GP'
        test = 'MS'

    # Split dataset based on school
    df_train = df[df['school'] == train].copy()
    df_test = df[df['school'] == test].copy()

    # The split is done using school as a criteria, so it becomes an unnecessary feature
    df_train.drop(columns=['school'], inplace=True)
    df_test.drop(columns=['school'], inplace=True)

    # Separate input features from target values for training
    X_train = df_train.drop(columns=['G3'], inplace=False)
    y_train = df_train['G3'].copy()

    # Separate input features from target values for testing
    X_test = df_test.drop(columns=['G3'], inplace=False)
    y_test = df_test['G3'].copy()


    return X_train, y_train, X_test, y_test

def evaluate_models(X_train, y_train, X_test, y_test, models, preprocessor, subject):
    """
    Iterates through models and evaluates them, while writing necessary stats and data to CSV files.

    Parameters:
        X_train (DataFrame): Training dataset of features.
        y_train (DataFrame): Training dataset of targets.
        X_test (DataFrame): Testing dataset of features.
        y_test (DataFrame): Testing dataset of targets.
        models (dict): List of models.
        preprocessor (ColumnTransformer): Column transformer which contains feature preprocessing methods.
        subject (string): School subject of data ("por" or "mat").
    """

    # For every model create individual pipelines
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)
        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Write important stats and results to CSV files
        calculate_and_write_results(name, pipeline, y_pred, y_test, subject)



def calculate_and_write_results(model_name, pipeline, y_pred, y_test, subject):
    """
    Writes model's coefficients, intercept and stats to CSV files.

    Parameters:
        model_name (string): Name of the model.
        pipeline (Pipeline): Pipeline utilized for this iteration.
        y_pred (DataFrame): Predicted values.
        y_test (DataFrame): True values.
        subject (string): School subject of data ("por" or "mat").
    """

    # Get categorical feature column headers
    preprocessor = pipeline.named_steps['preprocessor']
    ohe = preprocessor.named_transformers_['ohe']
    encoded_names = ohe.get_feature_names_out(categorical_features)

    # Recreate all feature column headers
    all_feature_names = list(encoded_names) + numeric_features
    # Also show intercept (bias) in the CSV
    all_feature_names.append('intercept')

    # Get coefficient (weights) and intercept (bias)
    model = pipeline.named_steps['regressor']
    weights = model.coef_
    intercept = model.intercept_

    # Round values to 4 decimals
    result = np.append(weights, intercept)
    rounded_result = np.round(result, decimals=4)

    result = pd.DataFrame([rounded_result], columns=all_feature_names)

    # Generate a prefix to tell different preprocessor combinations apart
    methods = []
    for name in preprocessor.named_transformers_.keys():
        methods.append(name)
    prefix = "_".join(methods) # More efficient than += operator

    result.to_csv(f"results/{subject}/{prefix}_{model_name}.csv", index=False, header=True)

    # Calculate stats
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Write to results/stats.csv
    add_stats(prefix, model_name, mae, rmse, r2)

def add_stats(prefix, model_name, mae, rmse, r2):
    """
    Write model's stats to results/stats.csv.

    Parameters:
        prefix (string): Prefix added to model's name.
        model_name (string): Name of the model.
        mae (float): Mean absolute error of the prediction.
        rmse (float): Root mean square error of the prediction.
        r2 (float): R^2 score of the prediction.
    """

    # Create a dataframe to append
    df = pd.DataFrame([[f"{prefix}_{model_name}", f"{mae:.2f}", f"{rmse:.2f}", f"{r2:.2f}"]],
                      columns=['name', 'mae', 'rmse', 'r2'])

    # Append dataframe to existing file
    df.to_csv(f"results/stats.csv", index=False, header=False, mode='a') 


def main():
    if __name__ == '__main__':
        data = read_csv("data/student-por.csv")

        X_train, y_train, X_test, y_test = split_data_for_lr(data)

        preprocessor = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_features),
                ('num', 'passthrough', numeric_features)
            ]
        )

        models = {
            'linear_regression': LinearRegression()
        }

        evaluate_models(X_train, y_train, X_test, y_test, models, preprocessor, 'por')



#maybe make everything a class, so there is no problem with feature names being global varialbes

categorical_features = ['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
numeric_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout',
                    'Dalc', 'Walc', 'health', 'absences']

main()





