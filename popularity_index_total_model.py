import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load the data from the CSV file
file_path = 'US_youtube_trending_data.csv'
data = pd.read_csv(file_path)
data = data[data['tags'] != '[None]']

# Set a seed and randomly select 5000 rows
np.random.seed(42)
random_indices = np.random.choice(data.index, size=5000, replace=False)
new_data = data.loc[random_indices]

# Extract the 'tags' column
tags_data = new_data['tags']

# Function to clean and vectorize the tags from a single row
def clean_and_vectorize_tags(tags):
    unique_tags = set(tags.split('|'))  # Split and remove duplicates
    sorted_tags = sorted(list(unique_tags))  # Sort
    return sorted_tags

# Apply the function to the 'tags' column of new_data
new_data['vectorized_tags'] = tags_data.apply(clean_and_vectorize_tags)

# Prepare text data for vectorization
new_data['joined_tags'] = new_data['vectorized_tags'].apply(lambda x: ' '.join(x))

# Using TfidfVectorizer
tfidf = TfidfVectorizer(tokenizer=lambda txt: txt.split())
tags_tfidf = tfidf.fit_transform(new_data['joined_tags'])

# Dimensionality Reduction
svd = TruncatedSVD(n_components=100)
tags_reduced = svd.fit_transform(tags_tfidf)

# Define the weights for the popularity index formula
delta, alpha, beta, gamma = 1, 3, 3, 0.5

# Function to calculate the Popularity Index (P)
def calculate_popularity_index(row, delta, alpha, beta, gamma):
    V = row['view_count']
    L = row['likes'] / (row['likes'] + row['dislikes'] + 1)
    D = row['dislikes'] / (row['likes'] + row['dislikes'] + 1)
    C = row['comment_count'] / (row['comment_count'] + 1)
    numerator = 10 * (delta * np.log10(V + 1) + alpha * L - beta * D + gamma * C)
    denominator = delta + alpha + beta + gamma
    return numerator / denominator

# Apply the function to compute P for each row
new_data['popularity_index'] = new_data.apply(lambda row: calculate_popularity_index(row, delta, alpha, beta, gamma), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tags_reduced, new_data['popularity_index'], test_size=0.2, random_state=42)

# Set up the Lasso and Ridge regression models
lasso_model = Lasso(alpha=0.1)
ridge_model = Ridge(alpha=0.1)

# Fit the models to the training data
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# Make predictions
y_pred_lasso_train = lasso_model.predict(X_train)
y_pred_lasso_test = lasso_model.predict(X_test)
y_pred_ridge_train = ridge_model.predict(X_train)
y_pred_ridge_test = ridge_model.predict(X_test)

# Calculate train and test error
lasso_train_error = mean_squared_error(y_train, y_pred_lasso_train)
lasso_test_error = mean_squared_error(y_test, y_pred_lasso_test)
ridge_train_error = mean_squared_error(y_train, y_pred_ridge_train)
ridge_test_error = mean_squared_error(y_test, y_pred_ridge_test)

# Output the errors
print(f"Lasso Train Error: {lasso_train_error}")
print(f"Lasso Test Error: {lasso_test_error}")
print(f"Ridge Train Error: {ridge_train_error}")
print(f"Ridge Test Error: {ridge_test_error}")

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Assuming df is your full dataframe after the previous preprocessing steps
# and tags_reduced is your feature set after TF-IDF vectorization and SVD dimensionality reduction

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tags_reduced, new_data['popularity_index'], test_size=0.2, random_state=42)

# Decision Tree Regression
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred_dt_train = decision_tree_model.predict(X_train)
y_pred_dt_test = decision_tree_model.predict(X_test)
dt_train_error = mean_squared_error(y_train, y_pred_dt_train)
dt_test_error = mean_squared_error(y_test, y_pred_dt_test)

# Random Forest Regression
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf_train = random_forest_model.predict(X_train)
y_pred_rf_test = random_forest_model.predict(X_test)
rf_train_error = mean_squared_error(y_train, y_pred_rf_train)
rf_test_error = mean_squared_error(y_test, y_pred_rf_test)

# K-Nearest Neighbors Regression
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
y_pred_knn_train = knn_model.predict(X_train)
y_pred_knn_test = knn_model.predict(X_test)
knn_train_error = mean_squared_error(y_train, y_pred_knn_train)
knn_test_error = mean_squared_error(y_test, y_pred_knn_test)

# Neural Networks (MLP Regressor)
mlp_model = MLPRegressor(random_state=42, max_iter=500)
mlp_model.fit(X_train, y_train)
y_pred_mlp_train = mlp_model.predict(X_train)
y_pred_mlp_test = mlp_model.predict(X_test)
mlp_train_error = mean_squared_error(y_train, y_pred_mlp_train)
mlp_test_error = mean_squared_error(y_test, y_pred_mlp_test)

# Print out the errors for each model
print(f"Decision Tree Train Error: {dt_train_error}")
print(f"Decision Tree Test Error: {dt_test_error}")
print(f"Random Forest Train Error: {rf_train_error}")
print(f"Random Forest Test Error: {rf_test_error}")
print(f"KNN Train Error: {knn_train_error}")
print(f"KNN Test Error: {knn_test_error}")
print(f"Neural Network Train Error: {mlp_train_error}")
print(f"Neural Network Test Error: {mlp_test_error}")
