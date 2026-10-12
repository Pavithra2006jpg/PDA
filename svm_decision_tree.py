import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)

# ---------- SVM (features are scaled first) ----------
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM accuracy:", round(accuracy_score(y_test, svm_pred), 4))
print(confusion_matrix(y_test, svm_pred))

# ---------- Decision Tree ----------
dt = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree accuracy:", round(accuracy_score(y_test, dt_pred), 4))
print(classification_report(y_test, dt_pred, target_names=wine.target_names))

print("5-fold CV  SVM :", round(cross_val_score(svm, X, y, cv=5).mean(), 4))
print("5-fold CV  Tree:", round(cross_val_score(dt, X, y, cv=5).mean(), 4))

# Graph 1: decision tree
plt.figure(figsize=(13, 6))
plot_tree(dt, feature_names=wine.feature_names, class_names=wine.target_names,
          filled=True, fontsize=8)
plt.savefig("q4_tree.png", dpi=130, bbox_inches="tight")
plt.show()

# Graph 2: SVM decision regions (2 features) and tree feature importance
f1, f2 = 6, 9    # flavanoids, color_intensity
svm2 = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))
svm2.fit(X[:, [f1, f2]], y)
xx, yy = np.meshgrid(np.linspace(X[:, f1].min() - .3, X[:, f1].max() + .3, 300),
                     np.linspace(X[:, f2].min() - .5, X[:, f2].max() + .5, 300))
zz = svm2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].contourf(xx, yy, zz, alpha=0.3, cmap="viridis")
ax[0].scatter(X[:, f1], X[:, f2], c=y, cmap="viridis", edgecolor="k", s=25)
ax[0].set_xlabel(wine.feature_names[f1])
ax[0].set_ylabel(wine.feature_names[f2])
ax[0].set_title("SVM (RBF) decision regions")

imp = dt.feature_importances_
top = np.argsort(imp)[-6:]
ax[1].barh([wine.feature_names[i] for i in top], imp[top], color="seagreen")
ax[1].set_title("Decision Tree feature importance")
plt.tight_layout()
plt.savefig("q4_svm_importance.png", dpi=130)
plt.show()
