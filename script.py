#! /usr/bin/python3

# ------------------------------------------------------------
# Name: Sabrina Akter
# Course: HSC 4933 - Python for Health Analytics
# Assignment: Lab 8
# Date: October 24, 2026
#
# Description:
# This program reads a dataset and lets the user choose
# predictor and response variables. The program can run
# logistic regression or Poisson regression. It also gives
# a confusion matrix for logistic regression and VIF values
# for Poisson regression. The results can be saved to a
# text file.
# ------------------------------------------------------------

import re
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Change a column name to snake_case
def to_snake_case(name):
    name = name.strip()
    name = re.sub(r'[^A-Za-z0-9]+', '_', name)
    name = name.lower()
    return name.strip('_')


# Ask user for dataset
FileName = input('Enter the dataset file name: ')

try:
    Data = pd.read_csv(FileName)
except FileNotFoundError:
    print('File not found.')
    raise SystemExit


# Clean column names
NewColumns = []

for Column in Data.columns:
    NewColumns.append(to_snake_case(Column))

Data.columns = NewColumns


# Show column names
print('\nAvailable columns:')

for Column in Data.columns:
    print(Column)


# Ask for predictors
PredictorInput = input(
    '\nEnter predictor column names separated by commas: '
)

Predictors = []

for Column in PredictorInput.split(','):
    Predictors.append(to_snake_case(Column))


# Ask for response
Response = input('Enter the response variable: ')
Response = to_snake_case(Response)


# Check that columns exist
SelectedColumns = Predictors + [Response]

for Column in SelectedColumns:
    if Column not in Data.columns:
        print('Column not found:', Column)
        raise SystemExit


# Remove rows with missing values
RegressionData = Data[SelectedColumns].dropna()

print('\nRows used:', len(RegressionData))


# Ask for regression type
print('\n1 - Logistic Regression')
print('2 - Poisson Regression')

Choice = input('Choose regression type [1/2]: ')


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------
if Choice == '1':

    X = RegressionData[Predictors]
    y = RegressionData[Response]

    # Check that response contains only 0 and 1
    Values = set(y.unique())

    if not Values.issubset({0, 1}):
        print('Response variable must contain only 0 and 1.')
        raise SystemExit

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Run logistic regression
    Model = LogisticRegression(max_iter=1000)
    Model.fit(X_train, y_train)

    Predictions = Model.predict(X_test)

    Accuracy = accuracy_score(y_test, Predictions)
    Report = classification_report(
        y_test,
        Predictions,
        zero_division=0
    )

    Matrix = confusion_matrix(y_test, Predictions)

    # Get pseudo R-squared
    X_stats = sm.add_constant(RegressionData[Predictors])

    StatsModel = sm.Logit(
        RegressionData[Response],
        X_stats
    ).fit(disp=False)

    PseudoR2 = StatsModel.prsquared


    Output = '\nLOGISTIC REGRESSION OUTPUT\n'
    Output += '----------------------------------------\n'
    Output += 'Predictors: ' + ', '.join(Predictors) + '\n'
    Output += 'Response: ' + Response + '\n\n'

    Output += 'Accuracy: ' + str(round(Accuracy, 4)) + '\n'
    Output += 'Pseudo R-squared: ' + str(round(PseudoR2, 4)) + '\n\n'

    Output += 'Classification Report:\n'
    Output += Report + '\n'

    Output += 'Confusion Matrix:\n'
    Output += str(Matrix) + '\n'

    print(Output)


# ------------------------------------------------------------
# Poisson Regression
# ------------------------------------------------------------
elif Choice == '2':

    Formula = Response + ' ~ ' + ' + '.join(Predictors)

    Model = smf.poisson(
        formula=Formula,
        data=RegressionData
    ).fit()

    Summary = Model.summary().as_text()


    # Calculate VIF
    X = RegressionData[Predictors]
    X = sm.add_constant(X)

    VIFList = []

    for Number in range(X.shape[1]):

        VIF = variance_inflation_factor(
            X.values,
            Number
        )

        VIFList.append(
            [X.columns[Number], VIF]
        )


    VIFTable = pd.DataFrame(
        VIFList,
        columns=['Variable', 'VIF']
    )


    Output = '\nCOUNT-VARIABLE REGRESSION OUTPUT\n'
    Output += '----------------------------------------\n'
    Output += 'Predictors: ' + ', '.join(Predictors) + '\n'
    Output += 'Response: ' + Response + '\n\n'

    Output += Summary + '\n\n'

    Output += 'VIF:\n'
    Output += VIFTable.to_string(index=False) + '\n'

    print(Output)


else:
    print('Invalid choice.')
    raise SystemExit


# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------
Save = input('\nWould you like to save the output? [Yes/No]: ')

if Save.lower() == 'yes' or Save.lower() == 'y':

    OutputName = input('Enter output file name: ')

    if not OutputName.endswith('.txt'):
        OutputName += '.txt'

    Outfile = open(OutputName, 'w')
    Outfile.write(Output)
    Outfile.close()

    print('Output saved to:', OutputName)