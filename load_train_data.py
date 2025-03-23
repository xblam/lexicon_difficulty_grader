import numpy as np
import pandas as pd
import os
import textwrap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


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


class BOWLogisticRegression:
    def __init__(self, max_iter=10000, test_size=0.2, random_state=42):
        self.model = LogisticRegression(max_iter=max_iter)
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        # Split data into train and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        # Fit model
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Predict on validation set
        y_pred = self.model.predict(self.X_val)
        acc = accuracy_score(self.y_val, y_pred)
        print("Validation accuracy:", acc)
        return acc

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)


if __name__ == '__main__':
    # read in the data
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    # DO THIS TO TEST WHEN YOU WANT TO TEST IT
    tr_text_list = x_train_df['text'].values.tolist()
    te_text_list = x_test_df['text'].values.tolist()


# MAKE THE VECTORIZER FOR THE XBOW-----------------------------------------------------------------------------------------------------------------
    bow = Vectorizer()
    X_train_vectorized = bow.fit_transform(tr_text_list)

    # bow.print_vocab_with_counts()

    print("Vocab size:", len(bow.get_vocab()))
    print("Shape of X_train_bow: ", X_train_vectorized.shape)

# RUN LR -------------------------------------------------------------------------------------------------------------------------------------
    # Run the encoder to get the y output values



    y_train = encoder(y_train_df['Coarse Label'].values)

    # Train + validate model
    lr_model = BOWLogisticRegression()
    lr_model.fit(X_train_vectorized, y_train)
    lr_model.evaluate()

    # Predict on test set
    X_test_vectorized = bow.transform(te_text_list)
    y_pred_test = lr_model.predict(X_test_vectorized)
    np.savetxt("yproba1_test.txt", y_pred_test, fmt='%d')




    # y_train = encoder(y_train_df['Coarse Label'].values)

    # print(y_train.shape)

    # # need to get the validation set

    # my_model = LogisticRegression(max_iter=1000)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_vectorized, y_train, test_size=0.2, random_state=42)
    # my_model.fit(X_train, y_train)
    # y_pred = my_model.predict(X_val)

    # print(y_pred.shape)
    
    # # np.savetxt("y_pred.txt", y_pred, fmt='%d')
    # # np.savetxt("y_val.txt", y_val, fmt='%d')

    # train_accuracy = accuracy_score(y_pred, y_val)
    # print("Train accuracy: ", train_accuracy)

    # # USED TO OUTPUT BEST GUESS INTO FILE FOR GRADESCOPE -----------------------------------------------------------------------------------------
    # X_test_vectorized = bow.transform(te_text_list)
    # y_pred = my_model.predict(X_test_vectorized)
    # print(y_pred.shape)
    # np.savetxt("yproba1_test.txt", y_pred, fmt='%d')





