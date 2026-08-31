import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

SESSIONS_DIR = os.path.join("output", "sessions")
RESULTS_DIR = os.path.join("output", "ml_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_all_sessions():
    files = glob.glob(os.path.join(SESSIONS_DIR, "*.csv"))
    if not files:
        print("No session files found. Run gui_main.py and label some data first.")
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            print(f"[skip] {f} is empty, skipping.")
            continue

        if "ground_truth" in df.columns:
            dfs.append(df)
        else:
            print(f"[skip] {f} has no 'ground_truth' column (recorded before labeling was added), skipping.")

    if not dfs:
        print("No labeled sessions found (missing 'ground_truth' column). Update gui_main.py and re-run a session.")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def evaluate_rule_based(df):
    """Your existing system's own call: DROWSY/CRITICAL counts as 'predicted drowsy'."""
    predicted = df["severity"].isin(["DROWSY", "CRITICAL"]).astype(int)
    actual = df["ground_truth"]
    return predicted, actual


def train_and_evaluate_ml(df):
    from sklearn.dummy import DummyClassifier

    features = ["ear", "mar", "perclos", "pitch", "fusion_score"]
    X = df[features]
    y = df["ground_truth"]

    if y.nunique() < 2:
        print("Ground truth labels only contain one class.")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)

    print(f"\n[sanity check] Dummy baseline (always predicts majority class):")
    print_metrics("Dummy Baseline", y_test, y_dummy)

    return y_test, y_pred, model, y_dummy



def plot_confusion(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Awake", "Drowsy"])
    ax.set_yticklabels(["Awake", "Drowsy"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    df = load_all_sessions()
    if df is None:
        return

    print(f"Loaded {len(df)} labeled frames from {df['ground_truth'].value_counts().to_dict()} distribution.")

    # ---------- Rule-based system evaluation ----------
    rb_pred, rb_actual = evaluate_rule_based(df)
    rb_metrics = print_metrics("Rule-Based Fusion System (your system)", rb_actual, rb_pred)
    plot_confusion(rb_actual, rb_pred, "Rule-Based System",
                    os.path.join(RESULTS_DIR, "confusion_rule_based.png"))

    # ---------- ML model evaluation ----------
    y_test, y_pred, model, y_dummy = train_and_evaluate_ml(df)
    if y_test is not None:
        ml_metrics = print_metrics("Logistic Regression (data-driven)", y_test, y_pred)
        plot_confusion(y_test, y_pred, "ML Classifier (Logistic Regression)",
                        os.path.join(RESULTS_DIR, "confusion_ml_model.png"))

        # ---------- Side-by-side comparison chart ----------
        metrics_names = ["accuracy", "precision", "recall", "f1"]
        rb_vals = [rb_metrics[m] for m in metrics_names]
        ml_vals = [ml_metrics[m] for m in metrics_names]

        x = np.arange(len(metrics_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(x - width/2, rb_vals, width, label="Rule-Based Fusion", color="#7b1fa2")
        ax.bar(x + width/2, ml_vals, width, label="ML (Logistic Regression)", color="#1976d2")
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics_names])
        ax.set_ylim(0, 1)
        ax.set_title("Rule-Based vs ML Classifier Performance")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "comparison_chart.png"), dpi=150)
        plt.close(fig)
        print(f"\nComparison chart saved to: {os.path.join(RESULTS_DIR, 'comparison_chart.png')}")

    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()