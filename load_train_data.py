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


# this function will turn the y values from the raw df into binaries with 0 represeting 2-3 and 1 representing 4-5
def encoder(y_raw):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    return y_encoded



if __name__ == '__main__':
    # read in the data
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    # DO THIS TO TEST WHEN YOU WANT TO TEST IT
    tr_text_list = x_train_df['text'].values.tolist()
    te_text_list = x_test_df['text'].values.tolist()


    # # print out he dimenssions of the data
    # N, n_cols = x_train_df.shape
    # print("Shape of x_train_df: ", (N, n_cols))
    # print("Shape of y_train_df: ", y_train_df.shape)
    # print("shape of x_test_df: ", x_test_df.shape)


# MAKE THE VECTORIZER FOR THE XBOW-----------------------------------------------------------------------------------------------------------------
    bow = Vectorizer()
    X_train_bow = bow.fit_transform(tr_text_list)

    X_test_bow = bow.transform(te_text_list)

    print("Vocab size:", len(bow.get_vocab()))
    print("Shape of X_train_bow: ", X_train_bow.shape)
    print("Shape of X_test_bow: ", X_test_bow.shape)

# RUN THE ENCODER-------------------------------------------------------------------------------------------------------------------------------------
    # Run the encoder to get the y output values
    y_true = encoder(y_train_df['Coarse Label'].values)

    print(y_true.shape)

    # need to get the validation set

    my_model = LogisticRegression(max_iter=10000)
    X_train, X_val, y_train, y_val = train_test_split(X_train_bow, y_true, test_size=0.2, random_state=42)
    my_model.fit(X_train, y_train)

    y_pred = my_model.predict(X_val)

    print(y_pred.shape)
    print(y_val.shape)

    # np.savetxt("y_pred.txt", y_pred, fmt='%d')
    # np.savetxt("y_val.txt", y_val, fmt='%d')

    train_accuracy = accuracy_score(y_pred, y_val)
    print("Train accuracy: ", train_accuracy)

    y_pred = my_model.predict(X_test_bow)
    print(y_pred.shape)
    np.savetxt("yproba1_test.txt", y_pred, fmt='%d')




