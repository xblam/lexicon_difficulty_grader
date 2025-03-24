import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import textwrap
from sklearn.feature_extraction.text import CountVectorizer


def lr_preprocess(dir_name):

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=None,
        min_df=5,
    )
    # read in the data
    data_dir = dir_name
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    tr_text_list = x_train_df['text'].values.tolist()
    te_text_list = x_test_df['text'].values.tolist()

    X_train_vectorized = vectorizer.fit_transform(tr_text_list)
    X_test_vectorized = vectorizer.transform(te_text_list)
    vocab = vectorizer.get_feature_names_out()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_df['Fine Label'].values)

    return X_train_vectorized, y_train, X_test_vectorized



def nn_preprocess(dir_name):
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=None,
        max_df=1.0,
        min_df=0
    )
    # read in the data
    data_dir = dir_name
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    tr_text_list = x_train_df['text'].values.tolist()
    te_text_list = x_test_df['text'].values.tolist()

    X_train_vectorized = vectorizer.fit_transform(tr_text_list)
    # think about extra things that you want to put here

    X_test_vectorized = vectorizer.transform(te_text_list)
    vocab = vectorizer.get_feature_names_out()

    label_encoder = LabelEncoder()

    # first things first we want to train this very specifically
    y_train = label_encoder.fit_transform(y_train_df['Coarse Label'].values)


    return X_train_vectorized, y_train, X_test_vectorized


if __name__ == "__main__":
    lr_preprocess('data_readinglevel')