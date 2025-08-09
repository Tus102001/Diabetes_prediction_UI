# app.py — Diabetes Risk Predictor + Model Insights
# -------------------------------------------------
# Requirements (install in your venv):
#   pip install streamlit pandas numpy scikit-learn joblib matplotlib
#
# Files expected:
#   models/model.pkl
#   models/feature_names.json
#   data/diabetes.csv   (same schema as training)
#
# Run:
#   streamlit run app.py

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# ---------- Page setup ----------
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Risk Predictor")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "model.pkl"
FEATURES_PATH = ROOT / "models" / "feature_names.json"
DATA_PATH = ROOT / "data" / "diabetes.csv"

# ---------- Helpers ----------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_names = json.load(open(FEATURES_PATH))
    return model, feature_names

def ensure_columns(df: pd.DataFrame, needed: list[str]) -> pd.DataFrame:
    """Reorder and ensure needed columns exist; raise helpful error if not."""
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    return df[needed]

def metrics_at_threshold(y_true, y_proba, thr: float):
    y_pred = (y_proba >= thr).astype(int)
    return (
        (y_pred == y_true).mean(),                                  # accuracy
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
        y_pred
    )

# ---------- Load model/feature list ----------
try:
    model, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts. {e}")
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.01)
    st.caption("Move right to reduce false positives; left to catch more positives.")
    st.markdown("---")
    st.caption("Tip: Use 'Model Insights' to see how metrics change with the threshold.")

# ---------- Tabs ----------
tab_pred, tab_insights = st.tabs(["🔮 Predict", "📈 Model Insights"])

# ============================
# 🔮 Predict tab
# ============================
with tab_pred:
    st.subheader("Enter Clinical Details")
    c1, c2, c3 = st.columns(3)
    inputs = {}

    with c1:
        inputs["Pregnancies"] = st.number_input("Pregnancies", 0, 20, 1, step=1)
        inputs["Glucose"] = st.number_input("Glucose", 0, 300, 120)
        inputs["BloodPressure"] = st.number_input("BloodPressure", 0, 200, 70)
    with c2:
        inputs["SkinThickness"] = st.number_input("SkinThickness", 0, 100, 20)
        inputs["Insulin"] = st.number_input("Insulin", 0, 900, 80)
        inputs["BMI"] = st.number_input("BMI", 0.0, 80.0, 32.0, step=0.1, format="%.1f")
    with c3:
        inputs["DiabetesPedigreeFunction"] = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5, step=0.01, format="%.2f")
        inputs["Age"] = st.number_input("Age", 1, 120, 33, step=1)

    if st.button("Predict"):
        X_user = pd.DataFrame([[inputs[c] for c in feature_names]], columns=feature_names)
        proba = float(model.predict_proba(X_user)[0, 1])
        pred = int(proba >= threshold)

        st.subheader("Result")
        cp, cc = st.columns(2)
        with cp:
            st.metric("Probability of Diabetes", f"{proba:.2%}")
        with cc:
            st.metric("Prediction", "Positive" if pred == 1 else "Negative")
        st.caption("This is a statistical estimate, not a medical diagnosis.")

# ============================
# 📈 Insights tab
# ============================
with tab_insights:
    st.subheader("Evaluation on Hold-out Test Split")

    if not DATA_PATH.exists():
        st.warning("Missing data/diabetes.csv – cannot compute insights.")
        st.stop()

    # Load and align data
    try:
        df = pd.read_csv(DATA_PATH)
        df = ensure_columns(df, feature_names + ["Outcome"])
    except Exception as e:
        st.error(f"Could not load/align data: {e}")
        st.stop()

    # Split to mimic training evaluation
    X = df[feature_names]
    y = df["Outcome"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Predict probabilities on test split
    y_proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, y_proba)
    acc, prec, rec, f1, y_pred = metrics_at_threshold(yte, y_proba, threshold)

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc:.2%}")
    m2.metric("ROC-AUC", f"{auc:.3f}")
    m3.metric("Threshold", f"{threshold:.2f}")

    st.caption("Metrics recomputed locally against a fresh test split to reflect your current environment.")

    # ---- ROC Curve
    st.write("### ROC Curve")
    fpr, tpr, _ = roc_curve(yte, y_proba)
    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    st.pyplot(fig1)

    # ---- Confusion Matrix (fixed to draw on our axis)
    st.write("### Confusion Matrix")
    cm = confusion_matrix(yte, y_pred)
    fig2, ax2 = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax2, values_format="d", colorbar=False)
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)

    # ---- Precision-Recall Curve
    st.write("### Precision–Recall Curve")
    precision, recall, _thr = precision_recall_curve(yte, y_proba)
    fig3, ax3 = plt.subplots()
    ax3.plot(recall, precision)
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision–Recall Curve")
    st.pyplot(fig3)

    # ---- Threshold vs Metrics (Accuracy / Precision / Recall / F1)
    st.write("### Threshold vs Metrics")
    thrs = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thrs:
        a, p, r, f, _ = metrics_at_threshold(yte, y_proba, t)
        rows.append((t, a, p, r, f))
    tdf = pd.DataFrame(rows, columns=["threshold","accuracy","precision","recall","f1"])

    fig4, ax4 = plt.subplots()
    ax4.plot(tdf["threshold"], tdf["accuracy"], label="Accuracy")
    ax4.plot(tdf["threshold"], tdf["precision"], label="Precision")
    ax4.plot(tdf["threshold"], tdf["recall"], label="Recall")
    ax4.plot(tdf["threshold"], tdf["f1"], label="F1")
    ax4.set_xlabel("Threshold")
    ax4.set_ylabel("Score")
    ax4.set_title("Metrics vs Threshold")
    ax4.legend()
    st.pyplot(fig4)

    # ---- Feature Importance (works for RandomForest or LogisticRegression)
    st.write("### Feature Importance")
    clf = getattr(model, "named_steps", {}).get("clf", None)
    importances = None
    labels = feature_names

    if clf is None:
        # model might be a plain estimator if not a pipeline
        clf = model

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        label_x = "Model Feature Importance"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        label_x = "Absolute Coefficients"
    else:
        importances = None

    if importances is not None:
        order = np.argsort(importances)[::-1]
        labels_sorted = [labels[i] for i in order]
        vals_sorted = np.asarray(importances)[order]

        fig5, ax5 = plt.subplots()
        ax5.barh(labels_sorted[::-1], vals_sorted[::-1])
        ax5.set_xlabel(label_x)
        ax5.set_ylabel("Features")
        ax5.set_title("Top Features")
        st.pyplot(fig5)
    else:
        st.info("Current classifier does not expose feature importance/coefficients.")

    st.markdown("---")
    st.caption("Adjust the threshold in the sidebar and watch the confusion matrix and metrics update.")
