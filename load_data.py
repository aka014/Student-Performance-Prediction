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

    # print("DataFrame head:")
    # print(df.head())
    # print("\nDataFrame info:")
    # df.info()

    # These features are obviously the most important ones, so the model will be tested without them
    if drop_grades:
        df.drop(columns=['G1', 'G2'], inplace=True)

    return df


def split_data_for_lr(df, train='GP'):
    """
    Splits dataset into training and test sets.

    Parameters:
        df (DataFrame): Dataset.
        train (str): 'GP' or 'MS'

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

    # print("DataFrame head:")
    # print(X_train.head())
    # print("\nDataFrame info:")
    # X_train.info()
    #
    # print("DataFrame head:")
    # print(X_test.head())
    # print("\nDataFrame info:")
    # X_test.info()
    #
    # print("DataFrame head:")
    # print(y_train.head())
    # print("\nDataFrame info:")
    # y_train.info()
    #
    # print("DataFrame head:")
    # print(y_test.head())
    # print("\nDataFrame info:")
    # y_test.info()

    return X_train, y_train, X_test, y_test

def evaluate_models(X_train, y_train, X_test, y_test, models, preprocessor, subject):
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        calculate_and_write_results(name, pipeline, y_pred, y_test, subject)



def calculate_and_write_results(model_name, pipeline, y_pred, y_test, subject):

    preprocessor = pipeline.named_steps['preprocessor']
    ohe = preprocessor.named_transformers_['ohe']
    encoded_names = ohe.get_feature_names_out(categorical_features)

    all_feature_names = list(encoded_names) + numeric_features
    # To also show bias in the csv
    all_feature_names.append('intercept (b)')


    print("Predictions:", y_pred)

    model = pipeline.named_steps['regressor']
    weights = model.coef_
    intercept = model.intercept_

    result = np.append(weights, intercept)
    rounded_result = np.round(result, decimals=4)

    result = pd.DataFrame([rounded_result], columns=all_feature_names)

    methods = []
    for name in preprocessor.named_transformers_.keys():
        methods.append(name)
    prefix = "_".join(methods) #creates a string efficiently

    result.to_csv(f"results/{subject}/{prefix}_{model_name}.csv", index=False, header=True)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    add_stats(prefix, model_name, mae, rmse, r2)

def add_stats(prefix, model_name, mae, rmse, r2):

    df = pd.DataFrame([[f"{prefix}_{model_name}", f"{mae:.2f}", f"{rmse:.2f}", f"{r2:.2f}"]], columns=['name', 'mae', 'rmse', 'r2'])
    df.to_csv(f"results/stats.csv", index=False, header=False, mode='a') #append to existing info


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





