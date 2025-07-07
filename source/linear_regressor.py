def evaluate_models_cont(X_train, y_train, X_test, y_test, models, preprocessor, subject):
    """
    Iterates through models and evaluates them, while writing necessary stats and data to CSV files.

    This function is called when preprocessors do not change number of feature columns after One-Hot Encoding
    (e.g. RFE and PolynomialFeatures).

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
            ('feature_selection', SelectKBest(score_func=f_regression, k=5)),  # Select the top 5 features
            ('regressor', model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)
        # Make predictions
        y_pred = pipeline.predict(X_test)



        # Write important stats and results to CSV files
        #calculate_and_write_results(name, pipeline, y_pred, y_test, subject)

        model = pipeline.named_steps['regressor']
        weights = model.coef_
        intercept = model.intercept_

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(weights)
        print(intercept)
        print(f"Mae: {mae}")
        print(f"RMSE: {rmse}")
        print(f"R2: {r2}")


def main():
    if __name__ == '__main__':
        data = read_csv("../data/student-por.csv")

        X_train, y_train, X_test, y_test = split_data_for_lr(data)

        preprocessor1 = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_features),
                ('pass', 'passthrough', numeric_features)
            ]
        )

        preprocessor2 = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_features),
                ('stand_scale', StandardScaler(), numeric_features)
            ]
        )



        models = {
            'linear_regression': LinearRegression()
        }

        # evaluate_models_cont(X_train, y_train, X_test, y_test, models, preprocessor1, 'por')
        evaluate_models_cont(X_train, y_train, X_test, y_test, models, preprocessor2, 'por')

