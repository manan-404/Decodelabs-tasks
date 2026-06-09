# Project 2: Data Classification Using AI
# Dataset: Iris | Algorithm: K-Nearest Neighbors (KNN)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd

# ─── 1. LOAD & UNDERSTAND THE DATASET ────────────────────────────────────────

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 55)
print(" IRIS DATASET — OVERVIEW")
print("=" * 55)
print(f"Total samples  : {len(df)}")
print(f"Features       : {list(iris.feature_names)}")
print(f"Classes        : {list(iris.target_names)}")
print(f"\nClass distribution:\n{df['species'].value_counts().to_string()}")
print(f"\nFirst 5 rows:\n{df.head().to_string()}")
print(f"\nBasic stats:\n{df.describe().to_string()}")

# ─── 2. FEATURE SCALING ──────────────────────────────────────────────────────

X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 55)
print(" AFTER STANDARD SCALING — first 3 rows")
print("=" * 55)
print(pd.DataFrame(X_scaled, columns=iris.feature_names).head(3).to_string())

# ─── 3. SPLIT INTO TRAINING & TESTING SETS ───────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,
    random_state=42,
    shuffle=True,
)

print("\n" + "=" * 55)
print(" TRAIN / TEST SPLIT")
print("=" * 55)
print(f"Training samples : {len(X_train)} (80%)")
print(f"Testing samples  : {len(X_test)}  (20%)")

# ─── 4. TRAIN KNN MODEL ──────────────────────────────────────────────────────

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("\n" + "=" * 55)
print(" MODEL TRAINED")
print("=" * 55)
print(f"Algorithm  : K-Nearest Neighbors")
print(f"K value    : {knn.n_neighbors}")

# ─── 5. PREDICT & VALIDATE ───────────────────────────────────────────────────

y_pred = knn.predict(X_test)

accuracy  = knn.score(X_test, y_test) * 100
f1        = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall    = recall_score(y_test, y_pred, average="weighted")

print("\n" + "=" * 55)
print(" VALIDATION RESULTS")
print("=" * 55)
print(f"Accuracy  : {accuracy:.2f}%")
print(f"F1 Score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

print("\nConfusion Matrix:")
print("(Rows = Actual, Cols = Predicted)")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {n}" for n in iris.target_names],
    columns=[f"Pred: {n}" for n in iris.target_names],
)
print(cm_df.to_string())

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
