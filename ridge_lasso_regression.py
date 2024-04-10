import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

datapath = './average_score.csv'
df = pd.read_csv(datapath)

# Define the features (independent variables) and the target (dependent variable)
X = df.drop("AVG_WIL", axis=1)  # Features
y = df["AVG_WIL"]  # Target

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the alpha parameters to test for Ridge and Lasso
alphas = np.logspace(-6, 6, 13)

# Initialize Ridge and Lasso with built-in cross-validation
ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=100000)

# Fit Ridge and Lasso models
ridge_model = ridge_cv.fit(X_train_scaled, y_train)
lasso_model = lasso_cv.fit(X_train_scaled, y_train)

# The best alpha and respective scores for Ridge
ridge_best_alpha = ridge_model.alpha_

# The best alpha and respective scores for Lasso
lasso_best_alpha = lasso_model.alpha_

# Predictions
ridge_predictions = ridge_model.predict(X_test_scaled)
lasso_predictions = lasso_model.predict(X_test_scaled)

# R-squared for test set
ridge_r2 = r2_score(y_test, ridge_predictions)
lasso_r2 = r2_score(y_test, lasso_predictions)

# RMSE for test set
ridge_rmse = sqrt(mean_squared_error(y_test, ridge_predictions))
lasso_rmse = sqrt(mean_squared_error(y_test, lasso_predictions))

# Combine the metrics into a dictionary
ridge_metrics = {'R-squared': ridge_r2, 'RMSE': ridge_rmse}
lasso_metrics = {'R-squared': lasso_r2, 'RMSE': lasso_rmse}

# Get the coefficients from the Ridge and Lasso models
ridge_coefficients = ridge_model.coef_
lasso_coefficients = lasso_model.coef_

# Output the results
ridge_output = {'best_alpha': ridge_best_alpha, 'rmse': ridge_rmse}
lasso_output = {'best_alpha': lasso_best_alpha, 'rmse': lasso_rmse}

# Print the results
print("Ridge Results:", ridge_output)
print("Lasso Results:", lasso_output)
print("Ridge Metrics:", ridge_metrics)
print("Lasso Metrics:", lasso_metrics)
print("Ridge_Coefficients:",ridge_coefficients)
print("Lasso_Coefficients:",lasso_coefficients)


# Actual vs Predicted for Ridge
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, ridge_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Ridge: Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Actual vs Predicted for Lasso
plt.subplot(1, 2, 2)
plt.scatter(y_test, lasso_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Lasso: Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()

