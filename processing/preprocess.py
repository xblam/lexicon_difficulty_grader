import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import textwrap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from .load_BERT_embeddings import load_BERT_embeddings


def lr_preprocess(dir_name='data_readinglevel', lowercase=True, stop_words=None, min_df=10, binary=True):

    # vectorizer = Vectorizer(
    #     lowercase=lowercase,
    #     stop_words=stop_words,
    #     max_df=max_df,
    #     min_df=min_df,
    # )
    
    vectorizer = TfidfVectorizer(
        lowercase=lowercase,
        stop_words=stop_words,
        min_df=min_df,
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
    # vocab = vectorizer.get_feature_names_out()

    label_encoder = LabelEncoder()

    if binary: y_train = label_encoder.fit_transform(y_train_df['Coarse Label'].values)
    else: y_train = label_encoder.fit_transform(y_train_df['Fine Label'].values)

    return X_train_vectorized, y_train, X_test_vectorized



def nn_preprocess(dir_name='data_readinglevel'):

    xBERT_train_NH, xBERT_test_NH = load_BERT_embeddings()
    xBERT_train_NH = pd.DataFrame(xBERT_train_NH)
    xBERT_test_NH = pd.DataFrame(xBERT_test_NH)
    print("bert loaded")

    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    numeric_features = [
    "char_count", "word_count", "sentence_count", "avg_word_length", "avg_sentence_length",
    "type_token_ratio", "pronoun_freq", "function_words_count", "punctuation_frequency",
    "sentiment_polarity", "sentiment_subjectivity", "readability_Kincaid", "readability_ARI",
    "readability_Coleman-Liau", "readability_FleschReadingEase", "readability_GunningFogIndex",
    "readability_LIX", "readability_SMOGIndex", "readability_RIX", "readability_DaleChallIndex",
    "info_characters_per_word", "info_syll_per_word", "info_words_per_sentence",
    "info_type_token_ratio", "info_characters", "info_syllables", "info_words", "info_wordtypes"
    ]
    x_handcrafted_train = x_train_df[numeric_features]
    x_handcrafted_test = x_test_df[numeric_features]

    # Combine BERT + other features
    X_train_combined = pd.concat([xBERT_train_NH, x_handcrafted_train], axis=1)
    X_test_combined = pd.concat([xBERT_test_NH, x_handcrafted_test], axis=1)

    print(X_train_combined.shape)
    print(X_train_combined.head())

    print(X_test_combined.shape)
    print(X_test_combined.head())

    y_train = LabelEncoder().fit_transform(y_train_df['Coarse Label'].values)
    print (set(y_train))

    return X_train_combined, y_train, X_test_combined


if __name__ == '__main__':
    nn_preprocess()