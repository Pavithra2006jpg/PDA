import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)

digits = load_digits()          # 1797 images of handwritten digits (8x8)
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=5,
    stratify=digits.target)

nb = GaussianNB(var_smoothing=1e-2)
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))
print("Predicted digit for test image 0:", nb.predict(X_test[:1])[0],
      "| actual:", y_test[0])

# Graphs: confusion matrix and per-digit recall
cm = confusion_matrix(y_test, y_pred)
recall = cm.diagonal() / cm.sum(axis=1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ConfusionMatrixDisplay(cm, display_labels=digits.target_names).plot(
    ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Naive Bayes confusion matrix")
ax[1].bar(digits.target_names, recall, color="slateblue")
ax[1].set_ylim(0, 1.05)
ax[1].set_xlabel("Digit")
ax[1].set_ylabel("Recall")
ax[1].set_title("Recall for each digit")
plt.tight_layout()
plt.savefig("q5_naive_bayes.png", dpi=130)
plt.show()
