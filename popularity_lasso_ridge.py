#ridge/lasso linear regression
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'alpha': np.logspace(-6, 6, 13)  # Or any other range/sequence of alpha values
}

# Set up the GridSearchCV object
ridge_grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters for Ridge from GridSearchCV:", ridge_grid_search.best_params_)
print("Best cross-validation score (MSE):", ridge_grid_search.best_score_)

# Evaluate on the test set
best_ridge_model = ridge_grid_search.best_estimator_
y_pred_test = best_ridge_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
print("Test MSE with the best Ridge model:", test_mse)
