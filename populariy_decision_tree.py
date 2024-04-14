###Decision Tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# Assuming 'tags_reduced' is the feature matrix from SVD and 'new_data' contains the target 'popularity_index'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tags_reduced, new_data['popularity_index'], test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor
decision_tree = DecisionTreeRegressor(random_state=42)

# Define a grid of hyperparameters to tune
param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
    'min_samples_split': [2, 10, 20],         # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 5, 10],           # Minimum number of samples required at a leaf node
    'max_features': ['auto', 'sqrt', 'log2', None]  # Number of features to consider when looking for the best split
}

# Setup the grid search to tune the hyperparameters
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters and best score from grid search
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

# Train a new decision tree using the best parameters found
optimized_decision_tree = DecisionTreeRegressor(**best_parameters, random_state=42)
optimized_decision_tree.fit(X_train, y_train)

# Predict on test data
y_pred_test = optimized_decision_tree.predict(X_test)

# Calculate test error
test_error = mean_squared_error(y_test, y_pred_test)

# Output the test error
print(f"Test Mean Squared Error: {test_error}")
