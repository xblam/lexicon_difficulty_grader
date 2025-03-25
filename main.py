import numpy as np
import pandas as pd
import os
from models import BOWLogisticRegressionCV
from processing import preprocess
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


if __name__ == '__main__':
    # in this case the y_train will be 4 classes
    X_train_vectorized, y_train, X_test_vectorized = preprocess(bianary=False)
    
    print(f'shape of x_train: {X_train_vectorized.shape}')
    print(f'set of y_train: {set(y_train)}')

    model = BOWLogisticRegressionCV(
        max_iter=10000, 
        test_size=0.2, 
        cv=5, 
        c_vals=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000],
        penalty=['l2'], 
        solver=['lbfgs'], 
        scorer='accuracy', 
        random_state=42)

    model.fit(X_train_vectorized, y_train)
    
    model.save_model()

    loaded_model = model.load_model()
    # acc = model.evaluate()

    # output = model.output_pred(X_test_vectorized)


    # np.savetxt("output/yproba1_test.txt", output, fmt='%.6f')


    # np.savetxt("output/yproba1_test.txt", model.predict(X_train_vectorized), fmt='%.6f')




# REGRESSION ON MULTIPLE CLASSES DONE, AND THEN WE CONCATENATE TOGETHER TO GET THE FINAL OUTPUT DECISION