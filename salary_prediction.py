import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
file_path = "employee_data.csv"
df = pd.read_csv(file_path)

# Step 1: Handle Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numerical missing values with median
df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical missing values with mode

# Step 2: Remove Outliers using IQR (Only for numeric columns)
numeric_cols = df.select_dtypes(include=['number']).columns  # Select numeric columns only
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 3: Convert categorical data to numeric using Label Encoding
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Step 4: Scale numerical values
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 5: Split dataset into training and testing sets
X = df.drop(columns=['Salary'])  # Features (Assuming 'Salary' is the target column)
y = df['Salary']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Development (Linear Regression and Random Forest Regression)

linear_model = LinearRegression()
rf_model = RandomForestRegressor()

linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Train Random Forest Model
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Step 7: Hyperparameter Tuning using GridSearchCV for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_rf_model = grid_search.best_estimator_

# Step 8: Model Evaluation

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Evaluating the Random Forest Model
y_pred_best_rf = best_rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_rf = r2_score(y_test, y_pred_best_rf)

# Display results
print("Linear Regression Model Evaluation:")
print(f"Mean Squared Error (Linear Regression): {mse_linear}")
print(f"R² Score (Linear Regression): {r2_linear}")

print("\nRandom Forest Model Evaluation (Before Hyperparameter Tuning):")
print(f"Mean Squared Error (Random Forest): {mse_rf}")
print(f"R² Score (Random Forest): {r2_rf}")

print(f"\nBest Hyperparameters from GridSearchCV: {grid_search.best_params_}")

# Final evaluation with the tuned model
y_pred_best_rf = best_rf_model.predict(X_test)
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

print("\nRandom Forest Model Evaluation (After Hyperparameter Tuning):")
print(f"Mean Squared Error (Best Random Forest): {mse_best_rf}")
print(f"R² Score (Best Random Forest): {r2_best_rf}")

# Display the training and testing set shapes
print(f"\nTraining Set Shape: {X_train.shape}")
print(f"Testing Set Shape: {X_test.shape}")
