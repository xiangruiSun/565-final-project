from experiment.model import *

def train(train_data, train_label, test_data, model_type):
    
    regression_model = Regression_Model(train_data, train_label, test_data)
    
    if model_type == "SVR": 
        test_pred = regression_model.SVR_model()
    elif model_type == 'LR':
        test_pred = regression_model.LinearRegression_model()
    elif model_type == 'GBR':
        test_pred = regression_model.GradientBoostingRegressor_model()
    elif model_type == 'EN':
        test_pred = regression_model.ElasticNet_model()
    elif model_type == 'optimized_EN':
        test_pred= regression_model.optimize_ElasticNet()
    elif model_type == 'KNN':
        test_pred= regression_model.KNN()
    elif model_type == 'NN_MLP':
        test_pred= regression_model.NN_MLP()
    elif model_type == 'RandomForest':
        test_pred= regression_model.RandomForest()
    elif model_type == 'DecisionTree':
        test_pred= regression_model.DecisionTree()
    
    return test_pred
    

def clustering(data):
    
    cluster = Cluster_Model(data)
    label, centroid = cluster.KMeans_model()
    return label, centroid