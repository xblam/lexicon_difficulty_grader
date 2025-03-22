''' Demo of how to load x/y data from CSV files

Will print out raw text and associated labels
for 8 randomly chosen examples in the training set.
'''



def print_random_rows():
    tr_text_list = x_train_df['text'].values.tolist()
    prng = np.random.RandomState(101)
    rows = prng.permutation(np.arange(y_train_df.shape[0]))
    for row_id in rows[:8]:
        text = tr_text_list[row_id]
        print("row %5d | %s BY %s | y = %s" % (
            row_id,
            y_train_df['title'].values[row_id],
            y_train_df['author'].values[row_id],
            y_train_df['Coarse Label'].values[row_id],
            ))
        # Pretty print text via textwrap library
        line_list = textwrap.wrap(tr_text_list[row_id],
            width=70,
            initial_indent='  ',
            subsequent_indent='  ')
        print('\n'.join(line_list))
        print("")

import numpy as np
import pandas as pd
import os
import textwrap

from sklearn.feature_extraction.text import CountVectorizer


if __name__ == '__main__':
    
    # read in the data
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
    tr_text_list = x_train_df['text'].values.tolist()

    # print out he dimenssions of the data
    N, n_cols = x_train_df.shape
    print("Shape of x_train_df: ", (N, n_cols))
    print("Shape of y_train_df: ", y_train_df.shape)
    print("shape of x_test_df: ", x_test_df.shape)


# MAKE THE VECTORIZER FOR THE BOW-----------------------------------------------------------------------------------------------------------------
    vectorizer = CountVectorizer(
        lowercase=True,
        # Rid of punctuations
        stop_words='english',
        # Will only exclude rare words since how often common words appear might gives clues to simplicity of article
        min_df=10
        )

    # Fit and transform (first run)
    BOWVec = vectorizer.fit_transform(tr_text_list)

    # Get vocab from the vectorizer, not the BOWVec
    vocab = vectorizer.get_feature_names_out()

    # Print vocab nicely
    print("Vocabulary:")
    for word in vocab:
        print(word)

    print(len(vocab))

