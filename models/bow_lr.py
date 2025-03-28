from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *

class BOWLogisticRegressionCV:
    def __init__(self, 
    max_iter=10000, 
    test_size=0.2, 
    cv=5, 
    c_vals=[1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3], 
    penalty=['l2'], 
    solver=['lbfgs'], 
    scoring='accuracy', 
    binary=True, 
    random_state=42):

        self.param_grid = {
            'C': c_vals,
            'penalty': penalty,
            'solver': solver
        }

        self.max_iter = max_iter
        self.test_size = test_size
        self.cv = cv
        self.random_state = random_state
        self.scoring = scoring
        self.binary = binary
        self.best_model = None

    def fit(self, X, y):
        # val set not used in cv folds, only for final eval
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        base_model = LogisticRegression(max_iter=self.max_iter)

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
        # get predicted probabilities
        self.y_preds = self.predict_proba(self.X_test)
        # print(f'Y_preds: {self.y_preds}')

        if not self.binary:
            print("MULTICLASS detected")

            # Convert y_test to binary
            y_true_binary = self.make_binary(self.y_test)

            # Get predicted class indices, then map to binary
            y_pred_classes = np.argmax(self.y_preds, axis=1)
            y_pred_binary = self.make_binary(y_pred_classes)

            # AUC and accuracy with multiclass -> binary
            auc = roc_auc_score(y_true_binary, y_pred_binary)
            acc = accuracy_score(y_true_binary, y_pred_binary)

            print(f"BINARY-FORM MULTICLASS Accuracy: {acc:.4f}")
            print(f"BINARY-FORM MULTICLASS AUC: {auc:.4f}")

        else:
            # bin classification
            y_probs = self.y_preds[:, 1]
            # print(f'Y_probs: {y_probs}')
            auc = roc_auc_score(self.y_test, y_probs)
            acc = accuracy_score(self.y_test, (y_probs >= 0.5).astype(int))

            print(f"BINARY Accuracy: {acc:.4f}")
            print(f"BINARY AUC: {auc:.4f}")

        return acc, auc


    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def predict_proba(self, X_test):
        return self.best_model.predict_proba(X_test)

    def save_model(self, path='output/lr_model.pkl'):
        model_bundle = {
            'model': self.best_model,
            'X_test': self.X_test,
            'y_test': self.y_test
        }
        with open(path, 'wb') as f:
            pickle.dump(model_bundle, f)
        print('model and validation set saved')

    def load_model(self, path='output/lr_model.pkl'):
        with open(path, 'rb') as f:
            model_bundle = pickle.load(f)

        self.best_model = model_bundle['model']
        self.X_test = model_bundle['X_test']
        self.y_test = model_bundle['y_test']

        print("model and validation set loaded successfully.")
        return self.best_model

    def make_binary(self, y):
        return np.where(y >= 2, 1, 0)

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

