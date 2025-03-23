array = [[23,  5],
       [ 3, 30]]

import seaborn as sns
from sklearn.metrics import confusion_matrix
sns.heatmap(array, annot=True)