"""Simple predictive models for storytelling: logistic regression and decision tree."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from analysis.common import build_player_feature_table, load_tables, prepare_output_dirs, setup_plot_style


def _feature_names(preprocessor: ColumnTransformer, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
    output_names = list(numeric_features)
    if categorical_features:
        encoder = preprocessor.named_transformers_["cat"]
        output_names.extend(list(encoder.get_feature_names_out(categorical_features)))
    return output_names


def run(processed_root: str | Path, output_root: str | Path) -> None:
    setup_plot_style()
    tables = load_tables(processed_root)
    feature_table = build_player_feature_table(
        players=tables["players"],
        votes=tables["votes"],
        speeches=tables["speeches"],
        night_actions=tables["night_actions"],
    )
    tables_dir, figures_dir = prepare_output_dirs(output_root, "05_modeling")

    model_df = feature_table.copy()
    model_df["won"] = model_df["won"].astype(int)

    numeric_features = [
        "survival_days",
        "day_votes_cast",
        "day_votes_received",
        "vote_accuracy",
        "vote_majority_alignment",
        "public_message_count",
        "public_word_count",
        "avg_words_per_message",
        "question_rate",
        "seer_hit_rate",
        "doctor_save_rate",
        "werewolf_consensus_rate",
    ]
    categorical_features = ["role"]

    keep_columns = ["won"] + numeric_features + categorical_features
    model_df = model_df[keep_columns].copy()

    X = model_df.drop(columns=["won"])
    y = model_df["won"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=405,
        stratify=y,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logistic = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    logistic.fit(X_train, y_train)
    logistic_pred = logistic.predict(X_test)
    logistic_proba = logistic.predict_proba(X_test)[:, 1]

    decision_tree = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=405)),
        ]
    )
    decision_tree.fit(X_train, y_train)
    tree_pred = decision_tree.predict(X_test)
    tree_proba = decision_tree.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "accuracy": accuracy_score(y_test, logistic_pred),
                "roc_auc": roc_auc_score(y_test, logistic_proba),
            },
            {
                "model": "decision_tree",
                "accuracy": accuracy_score(y_test, tree_pred),
                "roc_auc": roc_auc_score(y_test, tree_proba),
            },
        ]
    )
    metrics.to_csv(tables_dir / "model_metrics.csv", index=False)

    logistic_pre = logistic.named_steps["preprocess"]
    logistic_model = logistic.named_steps["model"]
    logistic_feature_names = _feature_names(logistic_pre, numeric_features, categorical_features)
    logistic_coef = pd.DataFrame(
        {
            "feature": logistic_feature_names,
            "coefficient": logistic_model.coef_[0],
        }
    ).sort_values("coefficient", key=lambda col: col.abs(), ascending=False)
    logistic_coef.to_csv(tables_dir / "logistic_coefficients.csv", index=False)

    top_coef = logistic_coef.head(15).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_coef["feature"], top_coef["coefficient"], color="#5E8C61")
    ax.set_title("Top Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    fig.savefig(figures_dir / "logistic_coefficients.png")
    plt.close(fig)

    tree_pre = decision_tree.named_steps["preprocess"]
    tree_model = decision_tree.named_steps["model"]
    tree_feature_names = _feature_names(tree_pre, numeric_features, categorical_features)
    tree_importance = pd.DataFrame(
        {
            "feature": tree_feature_names,
            "importance": tree_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    tree_importance.to_csv(tables_dir / "decision_tree_importance.csv", index=False)

    top_importance = tree_importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_importance["feature"], top_importance["importance"], color="#B95D4A")
    ax.set_title("Top Decision Tree Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.savefig(figures_dir / "decision_tree_importance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(
        tree_model,
        feature_names=tree_feature_names,
        class_names=["lose", "win"],
        filled=True,
        max_depth=3,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree (depth <= 3 shown)")
    fig.savefig(figures_dir / "decision_tree_structure.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run(processed_root=args.processed_root, output_root=args.output_root)


if __name__ == "__main__":
    main()
