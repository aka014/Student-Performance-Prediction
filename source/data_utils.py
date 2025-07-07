import pandas as pd

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


def write_result(result, model_name, subject):
    """
    Writes model's coefficients and intercept to CSV files.

    Parameters:
        result (DataFrame): DataFrame containing coefficients and intercept.
        model_name (string): Name of the model.
        subject (string): School subject of data ("por" or "mat").
    """


    result.to_csv(f"../results/{subject}/{model_name}.csv", index=False, header=True)


def add_stats(df, subject):
    """
    Write model's stats to results/{subject}/stats.csv.

    Parameters:
        df (DataFrame): DataFrame containing stats.
        subject (string): School subject of data ("por" or "mat").
    """

    df.to_csv(f"../results/{subject}/stats.csv", index=False, header=False, mode='a')







