from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt

# Simulate actual and predicted binary labels
actual = np.random.binomial(1, 0.9, size=1000)
predicted = np.random.binomial(1, 0.9, size=1000)

# Compute confusion matrix
confusion_matrix = confusion_matrix(actual, predicted)

# Display the matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
cm_display.plot()
plt.show()