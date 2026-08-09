import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

#Step 2: Load and Preprocess Data
# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Features and target
X = df.drop('MedHouseVal', axis=1)  # Features
y = df['MedHouseVal']  # Target (Median House Value)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )

# Feature scaling (important for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Step 3 Model Development
# Train a basic Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")

#Step 4 Hyperparameter Tuning
#1. Grid Search

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestRegressor(),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3,
                           verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Evaluate best model
y_pred = best_model.predict(X_test)
print(f"Tuned Model R²: {r2_score(y_test, y_pred)}")

#2. Random Search
from scipy.stats import randint

# Define parameter distribution
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=RandomForestRegressor(),
                                   param_distributions=param_dist,
                                   n_iter=50,
                                   scoring='neg_mean_squared_error',
                                   cv=3,
                                   verbose=2)

random_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", random_search.best_params_)

# Best model
best_model = random_search.best_estimator_

# Evaluate best model
y_pred = best_model.predict(X_test)
print(f"Tuned Model R²: {r2_score(y_test, y_pred)}")

#3.Full Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Pipeline for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), 
                          ('scaler', StandardScaler())]), X.columns)
    ]
)

# Full pipeline with model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Parameter grid for pipeline
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Grid Search with pipeline
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Evaluate
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)

print(f"Tuned Pipeline R²: {r2_score(y_test, y_pred)}")
