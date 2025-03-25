import numpy as np
import pandas as pd
import os
from models import BOWLogisticRegressionCV
from processing import preprocess
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pprint import pprint


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    # to run with just 2 classes, or to run with 4 classes then simplify to 2
    binary = True

    #PARAMS TO TWEAK HERE
    X_train_vectorized, y_train, X_test_vectorized = preprocess(
        dir_name='data_readinglevel', 
        lowercase=True, 
        stop_words='english', 
        max_df=.90, 
        min_df=0, 
        binary=binary,
    )

    # PARAMS TO TWEAK HERE
    model = BOWLogisticRegressionCV(
        max_iter=10000, 
        test_size=0.1, 
        cv=5, 
        # c_vals=np.logspace(-4, 4, 20),
        c_vals = [0.1],
        penalty=['l2'], 
        solver=['lbfgs'], 
        scorer='accuracy',
        binary=binary,
        random_state=96)
    
    # if you have trained and saved model comment out fit and run load
    model.fit(X_train_vectorized, y_train)
    # loaded_model = model.load_model()

    model.evaluate()
    model.save_model
    y_test_pred = model.predict(X_test_vectorized)

    # if not binary turn the guesses binary
    if not binary:
        y_test_pred = np.where(y_test_pred >= 2, 1, 0)

    print("OUTPUT PREDICTED")

    np.savetxt("output/yproba1_test.txt", y_test_pred, fmt='%.6f')
