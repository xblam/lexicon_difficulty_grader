import numpy as np
import pandas as pd
import os
from models import BOWLogisticRegressionCV, BOWNeuralNetworkCV
from processing import lr_preprocess, nn_preprocess
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pprint import pprint


# if __name__ == '__main__':
#     # to run with just 2 classes, or to run with 4 classes then simplify to 2
#     # binary = False
#     binary = True

#     #PARAMS TO TWEAK HERE
#     X_train_vectorized, y_train, X_test_vectorized = lr_preprocess(
#         dir_name='data_readinglevel',
#         lowercase=True,        
#         stop_words='english',
#         min_df=10,
#         binary=binary,
#     )

#     # PARAMS TO TWEAK HERE
#     model = BOWLogisticRegressionCV(
#         max_iter=10000,
#         test_size=0.1,
#         cv=5,
#         c_vals=[0.01, 0.05],
#         penalty=['l2'],
#         solver=['lbfgs'], 
#         # penalty=['elasticnet'],
#         # solver=['saga'],
#         scoring='accuracy',
#         binary=binary,
#         random_state=42
#     )

#     # model = BOWNeuralNetworkCV( 
#     #     max_iter=1000, 
#     #     test_size=0.1,
#     #     validation_fraction=0.1, 
#     #     cv=5, 
#     #     hidden_layer_sizes=[(100,)], 
#     #     activation='relu',
#     #     solver='adam',
#     #     alpha=[1e-4],
#     #     early_stopping=True,
#     #     n_iter_no_change=5,
#     #     batch_size='auto',
#     #     scoring='roc_auc',
#     #     binary=True, 
#     #     random_state=42
#     # )
    
#     # if you have trained and saved model comment out fit and run load
#     model.fit(X_train_vectorized, y_train)

#     model.evaluate()
#     model.save_model()
#     model.load_model()

#     y_test_pred = model.predict_proba(X_test_vectorized)

#     y_test_pred = y_test_pred[:, 1]

#     # if not binary turn the guesses binary
#     if not binary:
#         y_test_pred = np.where(y_test_pred >= 2, 1, 0)

#     print("OUTPUT PREDICTED")

#     np.savetxt("output/yproba1_test.txt", y_test_pred, fmt='%.6f')

#     # model.plot_confusion_matrix()

if __name__ == "__main__":
    x_train, y_train, x_test = nn_preprocess()


    model = BOWNeuralNetworkCV( 
        max_iter=1000, 
        test_size=0.1,
        validation_fraction=0.1, 
        cv=5, 
        hidden_layer_sizes=[(100,), (200,), (100, 100), (256, 128), (128, 64, 32)],
        activation=['relu', 'tanh', 'logistic'],
        solver=['adam'],
        alpha=[1e-4, 1e-3, 1e-2, 1e-1],
        learning_rate_init=[1e-4, 1e-3, 1e-2],
        batch_size=[32, 64, 128],
        scoring='roc_auc',
        early_stopping=True,
        n_iter_no_change=5,
        binary=True, 
        random_state=42
    )

    model.fit(x_train.values, y_train)
    model.evaluate()
    model.save_model()
    model.load_model()

    y_test_pred = model.predict_proba(x_test.values)[:,1]

    print("OUTPUT PREDICTED")

    np.savetxt("output/yproba1_test.txt", y_test_pred, fmt='%.6f')

    # model.plot_confusion_matrix()


    

