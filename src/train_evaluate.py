import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from generate_dataset import DATASET_PATH, generate_dataset


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "model_comparison.csv"
REPORT_PATH = RESULTS_DIR / "classification_reports.txt"
DATASET_SUMMARY_PATH = RESULTS_DIR / "dataset_summary.csv"


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", make_one_hot_encoder(), categorical_features),
        ]
    )


def build_models(preprocessor):
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.tight_layout()
    output_path = FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_comparison(metrics_df):
    melted = metrics_df.melt(
        id_vars="model",
        value_vars=["precision_macro", "recall_macro", "f1_macro"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Set")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)


def load_or_create_dataset():
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(DATASET_PATH, index=False)
    return dataset


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_or_create_dataset()
    dataset.describe(include="all").transpose().to_csv(DATASET_SUMMARY_PATH)

    target = "burnout_risk"
    drop_columns = ["employee_id", target]
    x = dataset.drop(columns=drop_columns)
    y = dataset[target]

    numeric_features = [
        "seniority_years",
        "total_weekly_hours",
        "overtime_hours",
        "daily_meeting_volume",
        "days_since_last_vacation",
        "sick_leave_days_6m",
        "self_reported_stress",
        "motivation_level",
    ]
    categorical_features = ["department", "role"]
    labels = ["Low", "Medium", "High"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor)

    metrics_rows = []
    report_blocks = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        metrics_rows.append(
            {
                "model": model_name,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
                "precision_weighted": weighted_precision,
                "recall_weighted": weighted_recall,
                "f1_weighted": weighted_f1,
            }
        )

        report_blocks.append(
            f"=== {model_name} ===\n"
            + classification_report(y_test, y_pred, labels=labels, zero_division=0)
        )
        plot_confusion_matrix(y_test, y_pred, labels, model_name)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1_macro", ascending=False)
    metrics_df.to_csv(METRICS_PATH, index=False)
    REPORT_PATH.write_text("\n\n".join(report_blocks), encoding="utf-8")
    plot_metric_comparison(metrics_df)

    print("Model comparison:")
    print(metrics_df.round(3).to_string(index=False))
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
