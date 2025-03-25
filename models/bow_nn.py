
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle


class BOWNeuralNetCV:
    def __init__(self, 
        max_iter=1000, 
        test_size=0.1,
        validation_fraction=0.1, 
        cv=5, 
        hidden_layer_sizes=[(100,)], 
        activation='relu',
        solver='adam',
        alpha=[1e-4],
        early_stopping=True,
        n_iter_no_change=5,
        batch_size='auto',
        scoring='roc_auc',
        binary=True, 
        random_state=42):

        # Store all parameters as instance variables
        self.max_iter = max_iter
        self.test_size = test_size
        self.validation_fraction = validation_fraction
        self.cv = cv
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.batch_size = batch_size
        self.scoring = scoring
        self.binary = binary
        self.random_state = random_state

        self.best_model = None
        self.grid_search = None

        self.param_grid = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': [self.activation],
            'alpha': self.alpha,
            'solver': [self.solver]
        }


    def fit(self, X, y):
        # val set not used in cv folds, only for final eval
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
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
            print(f"C = {params['C']:.4f} -> Mean CV Accuracy = {mean_score:.4f}")
        self.best_model = grid_search.best_estimator_
        print("Best CV Params:", grid_search.best_params_)


    def evaluate(self):
        # get the multi class mean test score on validation
        probs = self.predict_proba(self.X_val)[:,1]
        
        auc = roc_auc_score(self.y_val, probs)

        # if multiclass get the binary class test score on val
        if not self.binary:
            print(f'MULTICLASS multi val accuracy: {accuracy_score(self.y_val, y_preds)}')
            self.y_val = np.where(self.y_val >= 2, 1, 0)

        auc = roc_auc_score(self.y_val, probs)
        print(f"BINARY Validation Accuracy: {auc:.4f}")

        # return accuracy

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def predict_proba(self, X_test):
        return self.best_model.predict_proba(X_test)

    def save_model(self, path='output/lr_model.pkl'):
        model_bundle = {
            'model': self.best_model,
            'X_val': self.X_val,
            'y_val': self.y_val
        }
        with open(path, 'wb') as f:
            pickle.dump(model_bundle, f)
        print('model and validation set saved')

    def load_model(self, path='output/lr_model.pkl'):
        with open(path, 'rb') as f:
            model_bundle = pickle.load(f)

        self.best_model = model_bundle['model']
        self.X_val = model_bundle['X_val']
        self.y_val = model_bundle['y_val']

        print("model and validation set loaded successfully.")
        return self.best_model
