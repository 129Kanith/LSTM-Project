from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)

    report = classification_report(y_test, pred_classes)
    matrix = confusion_matrix(y_test, pred_classes)

    return report, matrix