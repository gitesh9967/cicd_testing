# -------------------------
# IMPORTS
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score

sns.set(style="whitegrid")


# -------------------------
# LOAD DATA
# -------------------------
dataset = pd.read_csv("iris.csv")

# Clean column names
dataset.columns = [
    col.strip().replace(" (cm)", "").replace(" ", "_")
    for col in dataset.columns
]


# -------------------------
# FEATURE ENGINEERING
# -------------------------
dataset["sepal_length_width_ratio"] = dataset["sepal_length"] / dataset["sepal_width"]
dataset["petal_length_width_ratio"] = dataset["petal_length"] / dataset["petal_width"]

dataset = dataset[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "sepal_length_width_ratio",
        "petal_length_width_ratio",
        "target",
    ]
]


# -------------------------
# TRAIN TEST SPLIT
# -------------------------
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=44)

X_train = train_data.drop("target", axis=1).astype("float32")
y_train = train_data["target"].astype("int32")

X_test = test_data.drop("target", axis=1).astype("float32")
y_test = test_data["target"].astype("int32")


# -------------------------
# LOGISTIC REGRESSION (UPDATED)
# -------------------------
logreg = LogisticRegression(
    C=0.0001,
    solver="lbfgs",
    max_iter=200  # slightly increased for stability
)

logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, pred_lr)
f1_lr = f1_score(y_test, pred_lr, average="micro")
prec_lr = precision_score(y_test, pred_lr, average="micro")
recall_lr = recall_score(y_test, pred_lr, average="micro")

train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100


# -------------------------
# RANDOM FOREST
# -------------------------
rf = RandomForestRegressor(random_state=44)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
pred_rf_class = np.round(pred_rf).astype(int)

f1_rf = f1_score(y_test, pred_rf_class, average="micro")
prec_rf = precision_score(y_test, pred_rf_class, average="micro")
recall_rf = recall_score(y_test, pred_rf_class, average="micro")

train_acc_rf = rf.score(X_train, y_train) * 100
test_acc_rf = rf.score(X_test, y_test) * 100


# -------------------------
# CONFUSION MATRIX PLOT
# -------------------------
def plot_confusion_matrix(cm, labels, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


plot_confusion_matrix(
    cm_lr,
    ["setosa", "versicolor", "virginica"],
    "ConfusionMatrix.png"
)


# -------------------------
# FEATURE IMPORTANCE
# -------------------------
importances = rf.feature_importances_
features = dataset.columns[:-1]

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(x="importance", y="feature", data=importance_df)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("FeatureImportance.png")
plt.close()


# -------------------------
# SAVE RESULTS
# -------------------------
with open("scores.txt", "w") as f:
    f.write("=== RANDOM FOREST ===\n")
    f.write(f"Train Accuracy: {train_acc_rf:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc_rf:.2f}%\n")
    f.write(f"F1 Score: {f1_rf:.4f}\n")
    f.write(f"Recall: {recall_rf:.4f}\n")
    f.write(f"Precision: {prec_rf:.4f}\n\n")

    f.write("=== LOGISTIC REGRESSION ===\n")
    f.write(f"Train Accuracy: {train_acc_lr:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc_lr:.2f}%\n")
    f.write(f"F1 Score: {f1_lr:.4f}\n")
    f.write(f"Recall: {recall_lr:.4f}\n")
    f.write(f"Precision: {prec_lr:.4f}\n")


print("✅ Training complete. Outputs saved:")
print("- ConfusionMatrix.png")
print("- FeatureImportance.png")
print("- scores.txt")
