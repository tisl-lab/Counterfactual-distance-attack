#written by meghana
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import json

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, activation):
        super(SimpleNet, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_layer_sizes[0]))
        self.activations.append(self.get_activation(activation))
        
        # Hidden layers
        for i in range(1, len(hidden_layer_sizes)):
            self.layers.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
            self.activations.append(self.get_activation(activation))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], 1))
        self.output_activation = nn.Sigmoid()  # Add this line to set the output activation function

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        x = self.output_activation(self.layers[-1](x))  # Use the output activation function
        return x

class SimpleNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_layer_sizes, activation, alpha, learning_rate, solver, epochs=30):
        self.model = SimpleNet(input_size, hidden_layer_sizes, activation)
        self.epochs = epochs
        self.alpha = alpha
        self.learning_rate = float(learning_rate)
        self.solver = solver
        self.criterion = nn.BCELoss()
        self.optimizer = self.get_optimizer()

    def get_optimizer(self):
        if self.solver == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        elif self.solver == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        else:
            raise ValueError(f"Unsupported solver: {self.solver}")

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
        
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
# class SimpleNetWrapper(BaseEstimator, ClassifierMixin):
#     def __init__(self, input_size, hidden_layer_sizes, activation, alpha, learning_rate, solver, epochs=10):
#         self.model = SimpleNet(input_size)
#         self.epochs = epochs
#         self.alpha = alpha
#         self.learning_rate = float(learning_rate)  # Ensure learning_rate is a float
#         self.solver = solver
#         self.criterion = nn.BCELoss()
#         self.optimizer = self.get_optimizer()

#     def get_optimizer(self):
#         if self.solver == 'adam':
#             return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
#         elif self.solver == 'sgd':
#             return optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
#         else:
#             raise ValueError(f"Unsupported solver: {self.solver}")

#     def fit(self, X, y):
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
#         for epoch in range(self.epochs):
#             self.model.train()
#             self.optimizer.zero_grad()
#             outputs = self.model(X_tensor)
#             loss = self.criterion(outputs, y_tensor)
#             loss.backward()
#             self.optimizer.step()
        
#         return self

#     def predict_proba(self, X):
#         self.model.eval()
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         with torch.no_grad():
#             outputs = self.model(X_tensor)
#         return outputs.numpy()

#     def predict(self, X):
#         proba = self.predict_proba(X)
#         return (proba > 0.5).astype(int)
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, activation):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_layer_sizes[0]))
        self.activations.append(self.get_activation(activation))
        
        # Hidden layers
        for i in range(1, len(hidden_layer_sizes)):
            self.layers.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
            self.activations.append(self.get_activation(activation))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], 1))
        self.activations.append(nn.Sigmoid())

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x

class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_layer_sizes, activation, alpha, learning_rate, solver, epochs=10):
        self.model = MLP(input_size, hidden_layer_sizes, activation)
        self.epochs = epochs
        self.alpha = alpha
        self.learning_rate = float(learning_rate)  # Ensure learning_rate is a float
        self.solver = solver
        self.criterion = nn.BCELoss()
        self.optimizer = self.get_optimizer()

    def get_optimizer(self):
        if self.solver == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        elif self.solver == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        else:
            raise ValueError(f"Unsupported solver: {self.solver}")

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
        
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

def calculate_input_size(numerical_features, categorical_features, X_train):
    # Create a ColumnTransformer to preprocess the data
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # Fit the preprocessor to the training data
    preprocessor.fit(X_train)
    
    # Calculate the input size after preprocessing
    num_numerical_features = len(numerical_features)
    num_categorical_features = sum(len(categories) for categories in preprocessor.named_transformers_['cat'].categories_)
    input_size = num_numerical_features + num_categorical_features
    
    return input_size

# def calculate_accuracy(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     return accuracy


def calculate_accuracy(pipeline, X_test, y_test):
    # Extract the PyTorch model from the pipeline
    model = pipeline.named_steps['NN'].model
    
    # Set the model to evaluation mode
    model.eval()
    
    # Transform the test data using the pipeline
    X_transformed = pipeline.named_steps['PP'].transform(X_test)
    
    # Convert the transformed test data to PyTorch tensors
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
    y_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    # Perform the forward pass and calculate the accuracy
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_tensor).float().mean().item()
    
    return accuracy


def create_model_pipeline(model_name, best_params, numerical_features, categorical_features, X_train):
    # Define the preprocessing steps in the ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Fit the preprocessor to the training data
    preprocessor.fit(X_train)

    # Calculate the number of categorical features after one-hot encoding
    num_categorical_features = sum(len(preprocessor.named_transformers_['cat'].categories_[i]) for i in range(len(categorical_features)))
    num_numerical_features = len(numerical_features)
    input_size = num_numerical_features + num_categorical_features

    # Define the model based on the provided model_name
    if model_name == 'RF':  # RandomForest
        model = RandomForestClassifier(**best_params)
    elif model_name == 'NN':
        hidden_layer_sizes = best_params['hidden_layer_sizes']
        activation = best_params['activation']
        alpha = best_params['alpha']
        learning_rate = best_params['learning_rate']
        solver = best_params['solver']
        epochs = best_params.get('epochs', 10)
        model = SimpleNetWrapper(input_size, hidden_layer_sizes, activation, alpha, learning_rate, solver, epochs)
    elif model_name == 'SVM':
        model = SVC(**best_params)
    elif model_name == 'XGBoost':
        model = XGBClassifier()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Create the pipeline
    pipeline = Pipeline([
        ('PP', preprocessor),
        (model_name, model)
    ])

    return pipeline

# def create_model_pipeline(model_name,best_params,numerical_features,categorical_features):
#     # Define your dataset and features
#     # Assuming DS.dataset['numerical_features'] and DS.dataset['categorical_features'] are defined
    
#     # Define the preprocessing steps in the ColumnTransformer
#     preprocessor = ColumnTransformer([
#         ('num', StandardScaler(), numerical_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ])

#     # Define the model based on the provided model_name
#     if model_name == 'RF': #randomforest
#         model = RandomForestClassifier(**best_params)
#     elif model_name == 'NN': 
#         model =  MLPClassifier(**best_params)#SimpleNet(input_dim)
#     elif model_name == 'SVM':
#         model = SVC(**best_params)
#     elif model_name == 'XGBoost':
#         model = XGBClassifier()
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")

#     # Create the pipeline
#     pipeline = Pipeline([
#         ('PP', preprocessor),
#         (model_name, model)
#     ])

#     return pipeline


def optimize_model_pipeline(model_name, numerical_features, categorical_features, X_train, y_train):
    # Create the model pipeline
    pipeline = create_model_pipeline(model_name, numerical_features, categorical_features)

    # Define the parameter grid for each model
    param_grid = {}
    if model_name == 'RF':
        param_grid = {
            'RF__n_estimators': [100], # [100, 200, 300],
            'RF__max_depth': [10] ,# [None, 10, 20, 30],
            'RF__min_samples_split': [2] # [2, 5, 10]
        }
    elif model_name == 'NN':
        param_grid = {
            'NN__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'NN__activation': ['relu', 'tanh'],
            'NN__solver': ['adam', 'sgd'],
            'NN__alpha': [0.0001, 0.001, 0.01],
            'NN__learning_rate': ['constant', 'adaptive']
        }
    elif model_name == 'SVM':
        param_grid = {
            'SVM__C': [0.1, 1, 10, 100],
            'SVM__gamma': [1, 0.1, 0.01, 0.001],
            'SVM__kernel': ['linear', 'rbf']
        }
    elif model_name == 'XGBoost':
        param_grid = {
            'XGBoost__n_estimators': [100, 200, 300],
            'XGBoost__learning_rate': [0.01, 0.1, 0.2],
            'XGBoost__max_depth': [3, 5, 7],
            'XGBoost__subsample': [0.8, 0.9, 1.0]
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"Best parameters for {model_name}: {best_params}")
    return best_model,best_params