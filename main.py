import numpy as np
import pandas as pd
import os
from models import BOWLogisticRegressionCV
from processing import nn_preprocess
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def run_lr():
    model = BOWLogisticRegressionCV(test_size=0.1)
    model.fit(X_train_vectorized, y_train)

    model.save_model()
    loaded_model = model.load_model()

    acc = model.evaluate()
    y_test_probs = loaded_model.predict_proba(X_test_vectorized)[:, 1]  # Only class 1 probs
    np.savetxt("output/yproba1_test.txt", y_test_probs, fmt='%.6f')

def run_nn():
    model = BOWNeuralNetworkCV(test_size=0.1)

    model.fit(X_train_vectorized, y_train)
    model.save_model()
    loaded_model = model.load_model()

    acc = model.evaluate()
    y_test_probs = loaded_model.predict_proba(X_test_vectorized)[:, 1]  # Only class 1 probs
    np.savetxt("output/yproba1_test.txt", y_test_probs, fmt='%.6f')

if __name__ == '__main__':
    # takes the raw files and outputes everything we need to run LR
    X_train_vectorized, y_train, X_test_vectorized = nn_preprocess('data_readinglevel')
    run_lr()

