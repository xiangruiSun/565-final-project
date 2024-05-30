from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import RidgeCV, LassoCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# import pandas as pd
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics import silhouette_score, davies_bouldin_score
# import matplotlib.pyplot as plt

# Step 1: Regression

class Regression_Model():
    
    def __init__(self, train_data, train_label, test_data):
        
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data

    def SVR_model(self):
        
        svr = SVR(kernel='rbf')
        svr.fit(self.train_data, self.train_label)
        test_pred = svr.predict(self.test_data)
        
        return test_pred

    def LinearRegression_model(self):
        
        lr = LinearRegression()
        lr.fit(self.train_data, self.train_label)
        test_pred = lr.predict(self.test_data)
        
        return test_pred

    def GradientBoostingRegressor_model(self):
        
        gbr = GradientBoostingRegressor()
        gbr.fit(self.train_data, self.train_label)
        test_pred = gbr.predict(self.test_data)
        
        return test_pred
    
    def ElasticNet_model(self, alpha=1.0, l1_ratio=0.5):

        enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        enet.fit(self.train_data, self.train_label)
        
        test_pred = enet.predict(self.test_data)
        
        return test_pred

    
    def optimize_ElasticNet(self):    
        model = ElasticNet()

        param_grid = {
            'alpha': np.logspace(-6, 6, 13)  # Or any other range/sequence of alpha values
        }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)

        grid_result = grid.fit(self.train_data, self.train_label)

        best_estimator = grid_result.best_estimator_
        
        test_pred = best_estimator.predict(self.test_data)
        return test_pred
    
    
    def KNN(self):
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
        grid_search.fit(self.train_data, self.train_label)
        # Best parameters found
        # print("Best parameters:", grid_search.best_params_)

        # Best score
        # print("Best cross-validation score (negative MSE):", grid_search.best_score_)

        # Evaluate on test set
        best_knn = grid_search.best_estimator_
        test_pred = best_knn.predict(self.test_data)

        return test_pred
    
    def NN_MLP(self):
        mlp = MLPRegressor(max_iter=1000, random_state=42)
        parameter_space = {
            'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate_init': [0.001, 0.01]
            }
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
        clf.fit(self.train_data, self.train_label)
        test_pred = clf.predict(self.test_data)

        return test_pred
    
    def RandomForest(self):
        # Initialize the Random Forest Regressor
        rf = RandomForestRegressor(random_state=42)

        # Define the hyperparameters to tune
        param_grid = {
            'n_estimators': [50, 100, 200],  # Number of trees
            'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],   # Minimum number of samples required at each leaf node
            'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
            }

        # Setup the GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

        # Fit the grid search to the data
        grid_search.fit(self.train_data, self.train_label)

        # Print the best parameters and best score from the grid search
        # print("Best parameters found: ", grid_search.best_params_)
        # print("Best cross-validation score: ", -grid_search.best_score_)

        # Retrieve the best model from the grid search
        best_rf_model = grid_search.best_estimator_

        # Optionally: Predict on the test set and calculate the test error
        test_pred = best_rf_model.predict(self.test_data)

        return test_pred
    
    def DecisionTree(self):
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
        grid_search.fit(self.train_data, self.train_label)

        # Best parameters and best score from grid search
        best_parameters = grid_search.best_params_
        best_score = grid_search.best_score_

        # Train a new decision tree using the best parameters found
        optimized_decision_tree = DecisionTreeRegressor(**best_parameters, random_state=42)
        optimized_decision_tree.fit(self.train_data, self.train_label)

        # Predict on test data
        test_pred = optimized_decision_tree.predict(self.test_data)

        return test_pred


# Step 2: Cluster

class Cluster_Model():
    
    def __init__(self, tags_vector):
        self.tags_vector = tags_vector
        
    
    def KMeans_model(self):
    
        kmeans = KMeans(n_clusters=2, random_state=0).fit(self.tags_vector)
        
        return kmeans.labels_, kmeans.cluster_centers_


    def other_cluster_method(self):
        pass
