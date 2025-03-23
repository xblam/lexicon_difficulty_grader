import numpy as np
import pandas as pd
import os
import textwrap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# vectorizes the text data to make the bow x dataset
class Vectorizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words='english',
            min_df=5,
        )

    def fit_transform(self, texts):
        self.X = self.vectorizer.fit_transform(texts)
        self.vocab = self.vectorizer.get_feature_names_out()
        return self.X

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def get_vocab(self):
        return self.vocab

    def print_vocab_with_counts(self):
        word_counts = self.X.sum(axis=0).A1  # Convert sparse matrix to flat array
        vocab = self.get_vocab()
        for word, count in zip(vocab, word_counts):
            print(f"{word}: {count}")


# this function will turn the y values from the raw df into binaries with 0 represeting 2-3 and 1 representing 4-5
def encoder(y_raw):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    return y_encoded


class BOWLogisticRegressionCV:
    def __init__(self, max_iter=10000, test_size=0.05, cv = 5, random_state=42):
        c_values = np.logspace(np.log10(1e-5), np.log10(1e5), num=100)
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
        # Split data into train and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        base_model = LogisticRegression(max_iter=10000)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            scoring='roc_auc', # so that it predicts using probabilities
            cv=self.cv,
            verbose=2, # Change depending on how detailed you want terminal to be
            n_jobs=-1 # Max cpu count and speed
        )

        grid_search.fit(self.X_train, self.y_train)

        # print out average accuracies for each level of c
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            print(f"C = {params['C']:.4f} -> Mean CV Accuracy = {mean_score:.4f}")
        self.best_model = grid_search.best_estimator_
        print("Best CV Params:", grid_search.best_params_)


    def evaluate(self):
        # Predict on validation set
        y_pred = self.best_model.predict(self.X_val)
        acc = accuracy_score(self.y_val, y_pred)
        print("Validation accuracy:", acc)
        return acc

    def predict_proba(self, X_test):
        return self.best_model.predict_proba(X_test)

    def get_best_params(self):
        return self.best_model.get_params()

    def save_model(self, filename):
        with open('bow_logistic_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)

    def load_model(self, filename):
        with open('bow_logistic_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def preprocess():
    # read in the data
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    tr_text_list = x_train_df['text'].values.tolist()
    te_text_list = x_test_df['text'].values.tolist()

    bow = Vectorizer()
    X_train_vectorized = bow.fit_transform(tr_text_list)

    y_train = encoder(y_train_df['Coarse Label'].values)

    X_test_vectorized = bow.transform(te_text_list)

    return X_train_vectorized, y_train, X_test_vectorized


def force_half_under_0_5(probas):
    """
    Given a 1D numpy array of probabilities, return a modified array
    where exactly half are below 0.5 and half are above, by ranking.
    """
    N = len(probas)
    sorted_indices = np.argsort(probas)
    
    # Create new probas initialized as 0.0
    new_probas = np.zeros(N)

    # First half (low confidence): assign evenly spaced values below 0.5
    for i, idx in enumerate(sorted_indices[:N//2]):
        new_probas[idx] = (i + 1) / (N + 1)  # Avoid exact 0

    # Second half (high confidence): assign values above 0.5
    for i, idx in enumerate(sorted_indices[N//2:]):
        new_probas[idx] = 0.5 + (i + 1) / (N + 1) * 0.5

    return new_probas


if __name__ == '__main__':
    # takes the raw files and outputes everything we need to run LR
    X_train_vectorized, y_train, X_test_vectorized = preprocess()

    model = BOWLogisticRegressionCV()

    model.fit(X_train_vectorized, y_train)
    acc = model.evaluate()
    model.save_model('best_model.pkl')

    loaded_model = model.load_model('best_model.pkl')

    y_test_probs = loaded_model.predict_proba(X_test_vectorized)[:, 1]  # Only class 1 probs

    np.savetxt("yproba1_test.txt", y_test_probs, fmt='%.6f')


    # new_probas = force_half_under_0_5(y_test_probs)
    # np.savetxt("yproba1_test.txt", new_probas, fmt='%.6f')



