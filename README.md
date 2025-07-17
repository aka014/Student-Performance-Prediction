# Predicting Student Performance using Linear and Polynomial Models

This repository contains the Python code for the project titled: "Predicting Student Performance using Linear and Polynomial Machine Learning Models: Comparison and Ethical Considerations."

## Project Overview

This project explores the use of linear regression models to predict students' final grades (G3) based on demographic, social, and academic features from the UCI Student Performance dataset. A key challenge of this work was to predict final grades *without* using prior-period grades (G1 and G2) and to test how well models trained on one school generalize to another.

The main findings include:
- Models trained on one school show poor generalization when tested on an unseen school, with the best models explaining only 5-10% of the variance in student grades.
- The number of past **failures** was consistently the most powerful negative predictor of academic success.
- For Portuguese students, the desire to pursue **higher education** was the strongest single positive predictor.

---

## Repository Structure

This repository contains the Python scripts used for the experiments.

-   `data_utils.py`: Utility functions for reading and splitting the dataset.
-   `features.py`: A class that defines the categorical and numerical features used in the models.
-   `dummy_regressor.py`: A baseline model that always predicts the mean.
-   `linear_regressor.py`: Implements the Ordinary Least Squares (OLS) regression model.
-   `ridge_regressor.py`: Implements Ridge Regression with Recursive Feature Elimination (RFE).
-   `lasso_regressor.py`: Implements Lasso Regression for feature selection.
-   `elastic_net.py`: Implements Elastic Net Regression.

---

## Dataset

The code in this repository uses the **Student Performance Data Set** from the UCI Machine Learning Repository. The dataset is not included in this repository.

1.  **Download the data**: You can download it from the [official UCI page](https://archive.ics.uci.edu/dataset/320/student+performance).
2.  **Placement**: Unzip the file and place `student-mat.csv` and `student-por.csv` into a `data/` directory in the root of this project.

---

## Requirements

The project requires Python 3 and the following libraries:

-   pandas
-   numpy
-   scikit-learn
-   matplotlib

You can install all required libraries using pip:
```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Usage

Each of the model scripts is designed to be run directly from the command line. To run an experiment, simply execute the desired Python script.

For example, to run the Lasso Regression experiment:
```bash
python lasso_regressor.py
```

The scripts will automatically read the data from the `data/` directory, run the experiments, and save the results (statistics and model coefficients) into a `results/` directory.

---

## Citation

If you use this code or the findings from the associated paper, please cite it.

**Full Paper:**
*A link to the full paper will be added here once it is available on Zenodo or another repository.*

**This Repository:**
```
```
