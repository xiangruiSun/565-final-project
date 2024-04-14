#KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error

knn = KNeighborsRegressor()

# Define the parameter grid
param_grid = {
    'n_neighbors': range(1, 31),
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
    'p': [1, 2]
}
# Use mean squared error as the scoring function
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Setup the grid search
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring=mse_scorer, verbose=1, n_jobs=-1)
# Fit grid search
grid_search.fit(X_train, y_train)
# Best parameters found
print("Best parameters:", grid_search.best_params_)

# Best score
print("Best cross-validation score (negative MSE):", grid_search.best_score_)

# Evaluate on test set
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", test_mse)

