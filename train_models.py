import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from sklearn.datasets import load_breast_cancer

# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()

# Select only the 10 mean features
selected_features = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension"
]
X = pd.DataFrame(data.data, columns=data.feature_names)[selected_features]
y = pd.Series(data.target)  # 0 = malignant, 1 = benign

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train SVM with imputer
# -----------------------------
svm_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("svc", SVC(probability=True))
])
svm_pipeline.fit(X_train, y_train)
joblib.dump(svm_pipeline, "models/svm.pkl")

# -----------------------------
# Train Random Forest with imputer
# -----------------------------
rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("rf", RandomForestClassifier(random_state=42))
])
rf_pipeline.fit(X_train, y_train)
joblib.dump(rf_pipeline, "models/random_forest.pkl")

# -----------------------------
# Extract and save feature importances
# -----------------------------
feature_names = X_train.columns
# ✅ Access the RandomForest step inside the pipeline
importances = rf_pipeline.named_steps["rf"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

importance_df.to_csv("models/random_forest_feature_importances.csv", index=False)

print("✅ Models trained and saved with imputer included")
print("📊 Random Forest feature importances saved to models/random_forest_feature_importances.csv")