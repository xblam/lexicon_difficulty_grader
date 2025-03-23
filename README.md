implementation details for q1:

Problem 1: Bag-of-Words Feature Representation
Background on Bag-of-Words Representations
As discussed in class on day10, the "Bag-of-Words" (BoW) representation assumes a fixed, finite-size vocabulary of V possible words is known in advance, with a defined index order (e.g. the first word is "stegosaurus", the second word is "dinosaur", etc.).

Each document is represented as a count vector of length V, where entry at index v gives the number of times that the vocabulary word with index v appears in the document.

The key constraint with BoW representations is that each input feature must directly correspond to one human-readable unigram in a finite vocabulary.

That said, you have many design decision to make when applying a BoW representation:

How big is your vocabulary?
* the vocab is 3235 words in total.

Do you exclude rare words (e.g. appearing in less than 10 documents)?
* yes we will, since this would just mean that the words are extremely uncommon, and we do not want really uncommon words to drastically shift the output of our model (even if this means that they most likely will also not appear that much in the testing sets)

Do you exclude common words (like 'the' or 'a', or appearing in more than 50% of documents)?
* no, we are not excluding common words, since the frequency of common words could actually give us a hint to how simple a passage is e.g passages that frequently use simple words might tend to be simpler in nature.

Do you keep the count values, or only store present/absent binary values?
* we will also be using the word count since as stated before, the frequency at which a word appears might help us determine how difficult the passage is.

User Guide for Bag of Words tools in sklearn.feature_extraction.text
sklearn.feature_extraction.text.CountVectorizer




1A : Bag-of-Words Design Decision Description
Well-written paragraph describing your chosen BoW feature representation pipeline, with sufficient detail that another student in this class could reproduce it. You are encouraged to use just plain English prose. You might include a brief, well-written pseudocode block if you think it is helpful.

You should describe and justify all major decisions, such as:

Did you perform any "cleaning" of the data? (e.g. handle upper vs. lower case, strip punctuation or unusual characters, etc.). Hint: don't spend too much time on cleaning. Simpler is better.
* for cleaning, we decided to keep it simple and remove all of the upper case and punctuation, and we also decided to get rid of stopwords


WORK FROM THIS POINT FORWARD____________________________________________________________________________________________________________-

How did you determine the final vocabulary set? Did you exclude any words? If so, why?
What was your final vocabulary size? If size varies across folds because it depends on the training set, please provide the typical size ranges.
Did you use counts or binary values or something else?
How does your approach handle out-of-vocabulary words in the test set? (you are allowed to just ignore them, but you should do this intentionally and your report should state clearly what is happening in your implementation and why).
Did you use off-the-shelf libraries? Or implement from scratch?
1B : Cross Validation Design Description
Well-written paragraph describing how you use cross-validation to perform both classifier training and any hyperparameter selection needed for the classifier pipeline.

For Problem 1, you must use cross validation with at least 3 folds, searching over at least 5 possible hyperparameter configurations to avoid overfitting.

You should describe and justify all major decisions, such as:

What performance metric will your search try to optimize on heldout data?
How many folds? How big is each fold?
Are folds just split at random? Did you you use other information to construct folds? Remember, your goal is to generalize from provided train set to provided test set.
After using CV to identify a selected hyperparameter configuration, how will you then build one "final" model to apply on the test set?
Did you use off-the-shelf libraries? Or implement from scratch?
1C : Hyperparameter Selection for Logistic Regression Classifier
For this step 1C, we want you to design and execute a hyperparameter search for a LogisticRegression classifier. Please use your BoW preprocessing from 1A and your CV design from 1B. Your CV procedure may only use the provided data in x_train.csv, y_train.csv. That procedure should allow you to estimate the heldout performance of a well-fit model with each candidate hyperparameter. Remember, your ultimate goal is to build a classifier pipeline that will achieve the best performance on the provided test set, as evaluated via Leaderboard submission later in 1E. But in this step, you only use estimates from CV.

In one paragraph about experimental design, you should describe and justify all major decisions, such as

Which hyperparameters are you searching?
What concrete grid of values will you try?
Next, you should include a figure that visualizes performance as a function of hyperparameter. Finally, you should include a caption paragraph summarizing the results of your hyperparameter search.

For all 3 parts above, please follow the detailed hyperparameter selection rubric which is common across Problem 1 and Problem 2.

1D : Analysis of Predictions for the Best Classifier
In a figure, provide a confusion matrix for the chosen best classifier from 1C. Be sure to look at heldout examples (not examples used to train that model). It's OK to analyze examples from just one fold (you don't need to look at all K folds throughout CV).

In a paragraph caption below the figure, try to characterize what kinds of mistakes the classifier makes.

We also encourage you to briefly try to understand the kinds of errors the classifier makes in more detail.

does it do better on longer or shorter documents?
does it do better on documents with less complex structure?
does it do better on a particular kind of author?
does it do better on some categories, like children's literature vs adult fiction?
You could use these insights to inform your open-ended work in Problem 2.

1E : Report Performance on Test Set via Leaderboard
Create your "final" classifier using the selected hyperparameters from 1C. Apply your classifier to each test sentence in x_test.csv. Store your probabilistic predictions into a single-column plain-text file yproba1_test.txt, as described above under What to Turn In. Upload this file to our Problem 1 leaderboard.

Remember, we'll use AUROC as the key performance metric that decides your rank on the leaderboard. Other metrics are also visible for information only.

In your report, include a summary paragraph stating your ultimate test set performance. Compare this value to your previous estimates of heldout performance from the results paragraph in 1C. Reflect on the numerical differences: do you think your search procedure gave an adequate estimate of heldout error? What might explain any differences?