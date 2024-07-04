# Covid_19_analysis

# COVID-19 Data Analysis and Mortality Rate Prediction

This repository contains data analysis and predictive modeling for COVID-19 cases using machine learning techniques.

## Overview

The project explores COVID-19 dataset containing global data on confirmed cases, deaths, recoveries, and active cases. The goal is to predict the mortality rate using regression models.

## Technologies Used

- **Python**: Programming language used for implementation
- **Pandas**, **NumPy**: Libraries for data manipulation and analysis
- **Seaborn**, **Matplotlib**: Libraries for data visualization
- **Scikit-learn**: Library for machine learning tasks
- **XGBoost**, **LightGBM**, **RandomForestRegressor**: Machine learning models used for prediction

## Data Preprocessing

- The dataset (`covid_19_clean_complete.csv`) is loaded and preprocessed:
  - Missing values in 'Province/State' are filled with 'Unknown'.
  - Missing 'Active' cases are filled with the difference between confirmed, deaths, and recovered.
  - Dates are converted to datetime format.

## Exploratory Data Analysis (EDA)

- Summary statistics and distribution plots are generated to understand the data:
  - Distribution of confirmed cases, deaths, recovered cases, and active cases.
  - Pairplot to visualize relationships between variables.
  - Time series analysis of COVID-19 cases over time.

## Feature Engineering

- New features such as Mortality Rate and Recovery Rate are computed.
- Scatter plots are used to visualize relationships between confirmed cases and mortality/recovery rates.

## Model Evaluation and Comparison

### Regression Models Evaluated:
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Support Vector Regressor (SVR)**
- **XGBoost Regressor**
- **LightGBM Regressor**

### Evaluation Metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R2)

### Results:

| Model                | MSE       | RMSE      | MAE       | R-squared |
|----------------------|-----------|-----------|-----------|-----------|
| Random Forest        | 0.002     | 0.045     | 0.028     | 0.905     |
| Gradient Boosting    | 0.003     | 0.053     | 0.032     | 0.873     |
| XGBoost              | 0.002     | 0.044     | 0.028     | 0.910     |
| LightGBM             | 0.002     | 0.044     | 0.028     | 0.911     |
| SVR                  | 0.010     | 0.100     | 0.056     | 0.606     |

## Conclusion

Based on the evaluation metrics, the **Random Forest Regressor** performs the best among the models tested for predicting the mortality rate of COVID-19 cases.

## Usage

To replicate the analysis and model training:

1. Clone this repository.
2. Install the required dependencies (`pip install -r requirements.txt`).
3. Ensure the dataset (`covid_19_clean_complete.csv`) is in the correct directory.
4. Run the Python scripts provided (`analysis.py`, `model_training.py`, etc.).

