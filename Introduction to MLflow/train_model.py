# Import libraries and modules
import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Training Data
df = pd.read_csv('Salary_predict.csv')
X = df[["experience", "age", "interview_score"]]
y = df[["Salary"]]


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

# Set Auto loggingfor scikit-learn flavor
mlflow.sklearn.autolog()
# Parameters
n_jobs_param = int(sys.argv[1])
fit_intercept_param = bool(sys.argv[2])

# Train model
lr = LinearRegression(n_jobs_param, fit_intercept=fit_intercept_param)
lr.fit(X_train, y_train)