
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

import numpy as np
import pickle


class BOWLogisticRegressionCV:
    def __init__(self, max_iter=20000, test_size=0.2, cv = 5, random_state=42):
        # c_values = np.logspace(np.log10(1e-3), np.log10(1e2), num=30)
        c_values = [0.1]
        self.param_grid = {
            'C': c_values,
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }

        self.model = LogisticRegression(max_iter=max_iter)
        self.test_size = test_size
        self.cv = cv
        self.random_state = random_state
        self.best_model = None


    def fit(self, X, y):
        # val set not used in cv folds, only for final eval
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        base_model = LogisticRegression(max_iter=10000)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            scoring='accuracy', # so that it predicts using probabilities
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
        # Get mean test score from the best CV result (already cross-validated)
        mean_score = self.grid_search.best_score_
        print(f"Mean CV: {mean_score:.4f}")

        # Combine the first two cols of X_val and y_val to get binary results
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
        with open('output/lr_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)

    def load_model(self):
        with open('output/lr_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model
