import numpy as np
import pandas as pd
import os
from models import BOWLogisticRegressionCV
from processing import nn_preprocess, lr_preprocess
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def run_lr():
    model = BOWLogisticRegressionCV(test_size=0.2)
    model.fit(X_train_vectorized, y_train)

    model.save_model()
    loaded_model = model.load_model()

    acc = model.evaluate()
    y_test_probs = loaded_model.predict_proba(X_test_vectorized)  # Only class 1 probs
    y_test_probs = y_test_probs[:,2] + y_test_probs[:,3] # Combine the last 2 to make our guess about which ones are difficult
    np.savetxt("output/yproba1_test.txt", y_test_probs, fmt='%.6f')


def run_nn():
    model = BOWNeuralNetworkCV(test_size=0.1)

    model.fit(X_train_vectorized, y_train)
    model.save_model()
    loaded_model = model.load_model()

    acc = model.evaluate()
    y_test_probs = loaded_model.predict_proba(X_test_vectorized)[:, 1] # Only class 1 probs
    np.savetxt("output/yproba1_test.txt", y_test_probs, fmt='%.6f')

if __name__ == '__main__':
    # takes the raw files and outputes everything we need to run LR
    X_train_vectorized, y_train, X_test_vectorized = lr_preprocess('data_readinglevel')
    model = BOWLogisticRegressionCV(test_size=0.1)

    model.fit(X_train_vectorized, y_train)
    model.save_model()

    loaded_model = model.load_model()
    acc = model.evaluate()

    output = model.output_pred(X_test_vectorized)


    np.savetxt("output/yproba1_test.txt", output, fmt='%.6f')


    # np.savetxt("output/yproba1_test.txt", model.predict(X_train_vectorized), fmt='%.6f')




# REGRESSION ON MULTIPLE CLASSES DONE, AND THEN WE CONCATENATE TOGETHER TO GET THE FINAL OUTPUT DECISION