import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Loading data
df = pd.read_csv("AirQuality.csv")
print(df.head())

# Identifying missing values
df.replace('NA', np.nan, inplace=True)
print(df.isnull().sum())

# Filling missing numerical values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Filling categorical missing values with mode
df['city'] = df['city'].fillna(df['city'].mode()[0])

# Handling duplicates
print("Duplicates before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates after:", df.duplicated().sum())

# Handling outliers
Q1 = df[['pollutant_min', 'pollutant_max', 'pollutant_avg']].quantile(0.25)
Q3 = df[['pollutant_min', 'pollutant_max', 'pollutant_avg']].quantile(0.75)
IQR = Q3 - Q1

# Removing extreme outliers
df = df[~((df[['pollutant_min', 'pollutant_max', 'pollutant_avg']] < (Q1 - 1.5 * IQR)) | (df[['pollutant_min', 'pollutant_max', 'pollutant_avg']] > (Q3 + 1.5 * IQR))).any(axis=1)]
df.head()

scaler = MinMaxScaler()
df[['pollutant_min', 'pollutant_max', 'pollutant_avg']] = scaler.fit_transform(df[['pollutant_min', 'pollutant_max', 'pollutant_avg']])

# Converting date column
df['last_update'] = pd.to_datetime(df['last_update'])
df.head()

def calculate_aqi(row):
    return (row['pollutant_avg'] * 100)

df['AQI'] = df.apply(calculate_aqi, axis=1)
df.head()

X = df[['pollutant_min', 'pollutant_max', 'pollutant_avg']]  # Feature set
y = df['AQI']  # Target variable

# Splitting data into training and testing sets (without using random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initializing and training a model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Hyperparameter tuning 
from sklearn.linear_model import Ridge

param_grid = {'alpha': [0.1, 1, 10, 100]}
ridge_model = Ridge()

grid_search = GridSearchCV(ridge_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameter from grid search
print(f"Best alpha parameter for Ridge regression: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"Mean Squared Error (Best Model): {mse_best}")
print(f"R² Score (Best Model): {r2_best}")
