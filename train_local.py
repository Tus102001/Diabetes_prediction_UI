import json, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data" / "diabetes.csv")

cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
        "BMI","DiabetesPedigreeFunction","Age","Outcome"]
df = df[cols].copy()
for c in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
    df[c] = df[c].replace(0, np.nan)

X = df.drop(columns=["Outcome"])
y = df["Outcome"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

num = X.columns.tolist()
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num)
])

pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
pipe.fit(Xtr, ytr)
auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:,1])
print(f"Saved LogisticRegression (test ROC-AUC={auc:.3f})")

(models_dir := ROOT / "models").mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, models_dir / "model.pkl")
with open(models_dir / "feature_names.json","w") as f:
    json.dump(num, f)
