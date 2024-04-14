#NN MLP
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Assuming previous steps from data loading to SVD are done and tags_reduced, new_data are ready
X_train, X_test, y_train, y_test = train_test_split(tags_reduced, new_data['popularity_index'], test_size=0.2, random_state=42)

# Set up the neural network and GridSearchCV
mlp = MLPRegressor(max_iter=1000, random_state=42)
parameter_space = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate_init': [0.001, 0.01]
}
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
test_error = mean_squared_error(y_test, y_pred)
print("Test Mean Squared Error:", test_error)
