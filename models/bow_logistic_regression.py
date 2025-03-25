
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle


class BOWLogisticRegressionCV:
    def __init__(self, max_iter=10000, test_size=0.2, cv=5, c_vals=0.1, penalty=['l2'], solver=['lbfgs'], scorer='accuracy', binary=True, random_state=42):
        self.param_grid = {
            'C': c_vals,
            'penalty': penalty,
            'solver': solver
        }

        self.max_iter = max_iter
        self.test_size = test_size
        self.cv = cv
        self.random_state = random_state
        self.scorer = scorer
        self.binary = binary
        self.best_model = None


    def fit(self, X, y):
        # val set not used in cv folds, only for final eval
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        base_model = LogisticRegression(max_iter=self.max_iter)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            scoring=self.scorer, # so that it predicts using probabilities
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
        y_preds = self.best_model.predict(self.X_val)
        print(y_preds.shape)

        print(self.binary)
        print(set(y_preds))
        print(set(self.y_val))

        # if multiclass get the binary class test score on val
        if not self.binary:
            print(f'MULTICLASS multi val accuracy: {accuracy_score(self.y_val, y_preds)}')
            print(self.y_val)
            print(set(self.y_val))
            print(y_preds)
            print(set(y_preds))
            y_preds_binary = np.where(y_preds >= 2, 1, 0)
            y_val_binary = np.where(self.y_val >= 2, 1, 0)
            accuracy = accuracy_score(y_val_binary, y_preds_binary)
            print(f"MULTICLASS Validation Accuracy: {accuracy:.4f}")
        else:
            accuracy = accuracy_score(self.y_val, y_preds)
            print(f"BINARY Validation Accuracy: {accuracy:.4f}")

        return accuracy

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def predict_proba(self, X_test):
        return self.best_model.predict_proba(X_test)

    def get_best_params(self):
        return self.best_model.get_params()

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
