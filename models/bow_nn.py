from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *

class BOWNeuralNetworkCV:
    def __init__(self, 
        max_iter=1000, 
        test_size=0.1,
        validation_fraction=0.1, 
        cv=5, 
        hidden_layer_sizes=[(100,)], 
        activation=['relu'],
        solver=['adam'],
        alpha=[1e-4],
        learning_rate_init=[1e-3],
        batch_size=['auto'],
        scoring='roc_auc',
        early_stopping=True,
        n_iter_no_change=5,
        binary=True, 
        random_state=42):

        # Store all parameters
        self.max_iter = max_iter
        self.test_size = test_size
        self.validation_fraction = validation_fraction
        self.cv = cv
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.scoring = scoring
        self.binary = binary
        self.random_state = random_state

        self.best_model = None
        self.grid_search = None

        self.param_grid = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'alpha': self.alpha,
            'learning_rate_init': self.learning_rate_init,
            'solver': self.solver,
            'batch_size': self.batch_size
        }
    def fit(self, X, y):
        # val set not used in cv folds, only for final eval
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        base_model = MLPClassifier(
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
            batch_size=self.batch_size
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            scoring=self.scoring, # so that it predicts using probabilities
            cv=self.cv,
            verbose=2, # Change depending on how detailed you want terminal to be
            n_jobs=-1 # Max cpu count and speed
        )

        self.grid_search = grid_search 
        grid_search.fit(self.X_train, self.y_train)

        # print out average accuracies for each level of c
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            print(f"Hidden layer sizers = {params['hidden_layer_sizes']} -> Mean CV Accuracy = {mean_score:.4f}")
        self.best_model = grid_search.best_estimator_
        print("Best CV Params:", grid_search.best_params_)


    def evaluate(self):
        probs = self.predict_proba(self.X_test)[:, 1]
        y_pred = (probs >= 0.5).astype(int)

        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, probs)

        print(f"Binary Accuracy: {acc:.4f}")
        print(f"Binary AUC: {auc:.4f}")
        return acc, auc

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def predict_proba(self, X_test):
        return self.best_model.predict_proba(X_test)

    def save_model(self, path='output/nn_model.pkl'):
        model_bundle = {
            'model': self.best_model,
            'X_test': self.X_test,
            'y_test': self.y_test
        }
        with open(path, 'wb') as f:
            pickle.dump(model_bundle, f)
        print('model and validation set saved')

    def load_model(self, path='output/nn_model.pkl'):
        with open(path, 'rb') as f:
            model_bundle = pickle.load(f)

        self.best_model = model_bundle['model']
        self.X_test = model_bundle['X_test']
        self.y_test = model_bundle['y_test']

        print("model and validation set loaded successfully.")
        return self.best_model

    def plot_confusion_matrix(self):
        actual = self.y_test
        predicted = self.best_model.predict(self.X_test)

        # Compute confusion matrix
        cm = confusion_matrix(actual, predicted)

        # Display the matrix with blue color map
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        cm_display.plot(cmap='Blues') 
        plt.title("Confusion Matrix")
        plt.show()

