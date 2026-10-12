import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)

# Sample data: study hours -> pass (1) / fail (0), 150 students
rng = np.random.default_rng(11)
hours = rng.uniform(0, 10, 150)
p_true = 1 / (1 + np.exp(-(1.2 * hours - 6)))
passed = (rng.random(150) < p_true).astype(int)
X = hours.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, passed, test_size=0.25, random_state=3, stratify=passed)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, prob)
print("AUC:", round(auc(fpr, tpr), 4))
print("P(pass) for 5 hours:", round(model.predict_proba([[5]])[0][1], 4))
print("Prediction for 8 hours:", model.predict([[8]])[0])

# Graphs: sigmoid curve and ROC curve
grid = np.linspace(0, 10, 200).reshape(-1, 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X_train, y_train, color="teal", alpha=0.6, label="Data")
ax[0].plot(grid, model.predict_proba(grid)[:, 1], color="crimson", label="Sigmoid")
ax[0].axhline(0.5, color="gray", linestyle="--")
ax[0].set_xlabel("Hours studied")
ax[0].set_ylabel("P(Pass)")
ax[0].set_title("Logistic Regression curve")
ax[0].legend()

ax[1].plot(fpr, tpr, color="darkorange", label=f"AUC = {auc(fpr, tpr):.2f}")
ax[1].plot([0, 1], [0, 1], "k--")
ax[1].set_xlabel("False positive rate")
ax[1].set_ylabel("True positive rate")
ax[1].set_title("ROC curve")
ax[1].legend()
plt.tight_layout()
plt.savefig("q3_logistic.png", dpi=130)
plt.show()
