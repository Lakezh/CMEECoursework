import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tensorflow.keras import models, layers  # Import libraries required for autoencoder


# Load DNA dataset
dna_df = pd.read_csv('dipeptide_data.csv')

# Select features and target variable
features_dna = dna_df.drop(columns=['Species', 'ogt'])
target_dna = dna_df['ogt']

# Handle missing values
features_dna = features_dna.dropna()
target_dna = target_dna.loc[features_dna.index]

# Standardize features
scaler_dna = StandardScaler()
features_dna_scaled = scaler_dna.fit_transform(features_dna)

# Flatten target variable if necessary
target_dna = target_dna.values.ravel()

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Bayesian Ridge': BayesianRidge(),
    'Elastic Net': ElasticNet(max_iter=800000, random_state=42),
    'Lasso': Lasso(max_iter=800000, tol=0.001, alpha=1.0, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Support Vector Regression (SVR)': SVR(),
    'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
    'XGBoost Regressor': xgb.XGBRegressor(random_state=42)
}

# Hyperparameter grids for tuning
param_grids = {
    'Elastic Net': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
    'Lasso': {'alpha': [0.01, 0.1, 1.0, 10.0]},
    'SVR': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20]},
    'Random Forest Regression': {'n_estimators': [50, 100, 200], 'max_features': ['sqrt', 'log2', None]},
    'Gradient Boosting Regressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'XGBoost Regressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
}

# Function to evaluate models with cross-validation and hyperparameter tuning
def evaluate_models(models, param_grids, features, target):
    results = {}
    best_model = None
    best_r2 = -float('inf')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        try:
            print(f"Evaluating model: {model_name}")
            if model_name in param_grids:
                grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(features, target)
                best_estimator = grid_search.best_estimator_
            else:
                best_estimator = model
                best_estimator.fit(features, target)

            cv_r2 = cross_val_score(best_estimator, features, target, scoring='r2', cv=kf, n_jobs=-1)
            cv_rmse = np.sqrt(-cross_val_score(best_estimator, features, target, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1))
            cv_mae = -cross_val_score(best_estimator, features, target, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)

            results[model_name] = {
                'Best Model': best_estimator,
                'CV R²': cv_r2.mean(),
                'CV RMSE': cv_rmse.mean(),
                'CV MAE': cv_mae.mean()
            }

            if cv_r2.mean() > best_r2:
                best_r2 = cv_r2.mean()
                best_model = best_estimator
        except Exception as e:
            print(f"Model {model_name} failed with error: {e}")

    return pd.DataFrame(results).T, best_model

# Evaluate models on DNA data
results, best_model = evaluate_models(models, param_grids, features_dna_scaled, target_dna)

# Display model results
print("Model Results:")
print(results)

# Plot results for the best model

def plot_cv_results(model, features, target, cv):
    # Use cross_val_predict to get predictions on the full dataset
    predictions = cross_val_predict(model, features, target, cv=cv)
    
    # Calculate R² and RMSE
    r_squared = r2_score(target, predictions)
    rmse = np.sqrt(np.mean((target - predictions) ** 2))
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with label for the legend
    plt.scatter(target, predictions, alpha=0.7, edgecolor='k', s=50, cmap='viridis', label='Predicted vs Actual')
    
    # Plot the diagonal line (perfect prediction)
    plt.plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=3, label='Perfect Prediction')
    
    # Add R² and RMSE to the plot
    plt.text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$\n$RMSE = {rmse:.2f}$',
             transform=plt.gca().transAxes, fontsize=18,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    
    # Axis labels and ticks
    plt.xlabel('Experimental growth temperature (°C)', fontsize=24)
    plt.ylabel('Predicted growth temperature (°C)', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    
    # Show plot
    plt.show()

# Example usage with the best model from GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
plot_cv_results(best_model, features_dna_scaled, target_dna, kf)
