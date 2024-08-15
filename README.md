# Titanic Survival Prediction Model 

![Python](https://img.shields.io/badge/python-3.6%2B-blue) ![Pandas](https://img.shields.io/badge/pandas-1.1.0%2B-orange) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.0%2B-yellowgreen)

This Python script implements a machine learning model to predict survival on the Titanic using the Titanic dataset. It employs Random Forest classifiers to generate predictions.

Find the Dataset here:
https://www.kaggle.com/competitions/titanic
## Features

- Random Forest model for prediction
- Processes the Titanic dataset
- Outputs predictions to a CSV file

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- numpy

Install the required packages with:

```bash
pip install pandas scikit-learn numpy
```
## Usage

Place train.csv and test.csv in the same directory as the script.

Run the script with:

```bash
python3 main.py
```
Predictions will be saved to prediction.csv.

## Output
The generated prediction.csv includes:

PassengerId: ID of the passenger

Survived: Predicted survival (0 = not survived, 1 = survived)
