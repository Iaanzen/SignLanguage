import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from code.create_dataset import custom_actions
from code.inference_classifier import model
from code.train_classifier import X_test, y_test
import matplotlib.pyplot as plt

y_true = [np.argmax(y) for y in y_test]
y_pred = [np.argmax(model.predict(np.expand_dims(x, axis=0))) for x in X_test]

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=custom_actions)
disp.plot(cmap='Blues')
plt.show()