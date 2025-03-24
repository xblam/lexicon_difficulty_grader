from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle


class BOWNeuralNetworkCV:
    def __init__(self, test_size=0.1, cv=5, random_state=42):
        self.param_grid = {
            'hidden_layer_sizes': [(500,), (100,100), (50, 50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [1e-2, 1e-3, 1e-4, 1e-5],
            'solver': ['adam']
        }
        self.test_size = test_size
        self.cv = cv
        self.random_state = random_state
        self.best_model = None


    def fit(self, X, y):
        # val set not used in cv folds, only for final eval
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        base_model = MLPClassifier(max_iter=300, random_state=self.random_state)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            scoring='accuracy',
            cv=self.cv,
            verbose=2,
            n_jobs=-1
        )

        self.grid_search = grid_search
        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_
        print("Best CV Params:", grid_search.best_params_)


        # print out average accuracies for each level of c
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            print(f"alpha = {params['alpha']:.4f} and layers = {params['alpha']:.4f} -> Mean CV Accuracy = {mean_score:.4f}")
        self.best_model = grid_search.best_estimator_
        print("Best CV Params:", grid_search.best_params_)
    

    def evaluate(self):
        # Get mean test score from the best CV result (already cross-validated)
        mean_score = self.grid_search.best_score_
        print(f"Mean CV: {mean_score:.4f}")

        val_score = self.best_model.score(self.X_val, self.y_val)
        print(f"Validation Accuracy: {val_score:.4f}")

        return val_score
        

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def predict_proba(self, X_test):
        return self.best_model.predict_proba(X_test)

    def get_best_params(self):
        return self.best_model.get_params()

    def save_model(self):
        with open('output/nn_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)

    def load_model(self):
        with open('output/nn_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model