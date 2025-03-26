train_set_labels = [
    "author",
    "title",
    "passage_id",
    "text",
    "char_count",
    "word_count",
    "sentence_count",
    "avg_word_length",
    "avg_sentence_length",
    "type_token_ratio",
    "pronoun_freq",
    "function_words_count",
    "punctuation_frequency",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "readability_Kincaid",
    "readability_ARI",
    "readability_Coleman-Liau",
    "readability_FleschReadingEase",
    "readability_GunningFogIndex",
    "readability_LIX",
    "readability_SMOGIndex",
    "readability_RIX",
    "readability_DaleChallIndex",
    "info_characters_per_word",
    "info_syll_per_word",
    "info_words_per_sentence",
    "info_type_token_ratio",
    "info_characters",
    "info_syllables",
    "info_words",
    "info_wordtypes"
]

test_set_labels = [
    "author",
    "title",
    "passage_id",
    "Coarse Label",
    "Fine Label"
]


import os

# Create folder if it doesn't exist

# Path to output file
file_path = os.path.join("data_readinglevel", "yproba1_test.txt")

# # Write 1197 lines of 0.1
# with open(file_path, "w") as f:
#     for _ in range(1197):
#         f.write("0.1\n")