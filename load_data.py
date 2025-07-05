import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
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

def encode_categorical_features(features):
    """

    """


def split_data_for_lr(df, train='GP', test='MS'):
    """
    First do this split and then create pipelines where i also might not even need this encoding function
    we will go with school a train and school b test since there are no hyperparameters
    """

    df_train = df[df['school'] == train].copy()
    df_test = df[df['school'] == test].copy()

    # Separate input features from target values for training
    X_train = df_train.drop(columns=['G3'], inplace=False)
    y_train = df_train['G3'].copy()

    # Separate input features from target values for testing
    X_test = df_test.drop(columns=['G3'], inplace=False)
    y_test = df_test['G3'].copy()

    print("DataFrame head:")
    print(X_train.head())
    print("\nDataFrame info:")
    X_train.info()

    print("DataFrame head:")
    print(X_test.head())
    print("\nDataFrame info:")
    X_test.info()

    print("DataFrame head:")
    print(y_train.head())
    print("\nDataFrame info:")
    y_train.info()

    print("DataFrame head:")
    print(y_test.head())
    print("\nDataFrame info:")
    y_test.info()

    return X_train, y_train, X_test, y_test



data = read_csv("data/student-por.csv")

X_train, y_train, X_test, y_test = split_data_for_lr(data)

categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']






# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")



# # --- 2. Create the Linear Regression model ---
# model = LinearRegression()
#
# # --- 3. Train (fit) the model using the training data ---
# print("\nTraining the Linear Regression model...")
# model.fit(X_train, y_train)
# print("Model training complete. 🎉")
#
# # --- 4. Make predictions on the test data ---
# y_pred = model.predict(X_test)
# print("\nPredictions on the test set generated. 📊")
#
# # --- 5. Evaluate the model ---
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse) # Root Mean Squared Error
# r2 = r2_score(y_test, y_pred)
#
# print(f"\n--- Model Evaluation ---")
# #print(model.score(X_test, y_test))
# print(f"Model Intercept (b0): {model.intercept_:.2f}")
# # if model.coef_.ndim > 1: # Check if coefficients are nested (for multiple features)
# #     print(f"Model Coefficients (b1, b2, ...): {np.array(model.coef_).flatten()}")
# # else:
# #     print(f"Model Coefficient (b1): {model.coef_:.2f}")
#
# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
# print(f"R-squared (R2) Score: {r2:.2f}")
#
# print(y_test)
# dataf = pd.DataFrame(y_pred)
# print(dataf)