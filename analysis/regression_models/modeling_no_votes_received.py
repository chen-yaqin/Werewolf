"""
Werewolf Project — Regression Models (votes_received excluded)
Removes n_votes_received / n_day_votes_received / n_night_votes_received
from all feature sets to isolate the predictive power of active behaviors
(votes cast, speech, role actions) independent of how much others targeted you.
Outputs → outputs_/plots/  and  outputs_/tables/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
SPEECH_CSV = os.path.join(HERE, "../speech_analysis/Outputs/tables/speech_features_by_player.csv")
VOTE_CSV   = os.path.join(HERE, "../vote_analysis/outputs/vote_features_by_player.csv")
ROLE_CSV   = os.path.join(HERE, "../role_analysis/outputs/role_features_by_player.csv")
OUT_PLOTS  = os.path.join(HERE, "outputs_/plots")
OUT_TABLES = os.path.join(HERE, "outputs_/tables")
os.makedirs(OUT_PLOTS,  exist_ok=True)
os.makedirs(OUT_TABLES, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
PALETTE     = ["#a9a7c7", "#e8cc7a", "#e48375", "#7fb8b0"]
ROLE_COLORS = {"Villager": "#e8cc7a", "Werewolf": "#7fb8b0",
               "Seer": "#a9a7c7", "Doctor": "#e48375"}
sns.set_theme(style="whitegrid", font_scale=1.05)

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD & MERGE
# ══════════════════════════════════════════════════════════════════════
speech = pd.read_csv(SPEECH_CSV)
vote   = pd.read_csv(VOTE_CSV)
role   = pd.read_csv(ROLE_CSV)

_shared = ["role", "model_name", "alive_end", "eliminated_during_day",
           "eliminated_during_phase", "winner_team", "last_day", "n_players", "end_reason"]

df = (speech
      .merge(vote.drop(columns=_shared), on=["game_id", "player_id"])
      .merge(role[["game_id", "player_id",
                   "n_inspects", "inspect_success_rate",
                   "n_heals", "heal_success_rate",
                   "n_wolf_votes", "wolf_day_consistency_rate"]],
             on=["game_id", "player_id"]))

df = df[df["winner_team"].isin(["Villagers", "Werewolves"])].copy()

df["survive"]    = df["alive_end"].astype(int)
df["player_win"] = (
    ((df["role"] == "Werewolf") & (df["winner_team"] == "Werewolves")) |
    ((df["role"] != "Werewolf") & (df["winner_team"] == "Villagers"))
).astype(int)

print(f"Dataset: {len(df)} rows  |  "
      f"survive rate: {df['survive'].mean():.3f}  |  "
      f"team-win rate: {df['player_win'].mean():.3f}")

# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE SETS  (votes_received columns excluded)
# ══════════════════════════════════════════════════════════════════════
CAT_COLS    = ["role", "model_name"]
# Removed: n_votes_received, n_day_votes_received, n_night_votes_received
VOTE_COLS   = ["n_votes_cast", "n_day_votes_cast", "n_night_votes_cast"]
SPEECH_COLS = ["n_messages", "avg_text_len", "first_day_messages", "first_day_text_len"]
ROLE_COLS   = ["n_inspects", "inspect_success_rate",
               "n_heals", "heal_success_rate",
               "n_wolf_votes", "wolf_day_consistency_rate"]
ALL_COLS    = CAT_COLS + VOTE_COLS + SPEECH_COLS + ROLE_COLS

INCR_MODELS = {
    "Model 1\n(role+model)":   CAT_COLS,
    "Model 2\n(+vote cast)":   CAT_COLS + VOTE_COLS,
    "Model 3\n(+speech)":      CAT_COLS + VOTE_COLS + SPEECH_COLS,
    "Model 4\n(+role action)": ALL_COLS,
}

ABLA_MODELS = {
    "Baseline\n(role+model)": CAT_COLS,
    "Vote cast\nonly":        CAT_COLS + VOTE_COLS,
    "Speech\nonly":           CAT_COLS + SPEECH_COLS,
    "Role action\nonly":      CAT_COLS + ROLE_COLS,
    "All features\n(Model 4)":ALL_COLS,
}

TARGETS = [
    ("survive",    "Player Survival"),
    ("player_win", "Player's Team Win"),
]

# ── Pipeline helpers ───────────────────────────────────────────────────
def _preprocessor(feat_cols):
    cat_cols = [c for c in feat_cols if c in CAT_COLS]
    num_cols = [c for c in feat_cols if c not in CAT_COLS]
    steps = []
    if cat_cols:
        steps.append(("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols))
    if num_cols:
        steps.append(("num", StandardScaler(), num_cols))
    return ColumnTransformer(steps, remainder="drop")

def lr_pipe(feat_cols):
    return Pipeline([("prep", _preprocessor(feat_cols)),
                     ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))])

def dt_pipe(feat_cols, max_depth=3):
    cat_cols = [c for c in feat_cols if c in CAT_COLS]
    num_cols = [c for c in feat_cols if c not in CAT_COLS]
    steps = []
    if cat_cols:
        steps.append(("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols))
    if num_cols:
        steps.append(("num", "passthrough", num_cols))
    prep = ColumnTransformer(steps, remainder="drop")
    return Pipeline([("prep", prep),
                     ("clf", DecisionTreeClassifier(max_depth=max_depth,
                                                    min_samples_leaf=50,
                                                    random_state=42))])

def get_feature_names(pipe, feat_cols):
    ct = pipe.named_steps["prep"]
    names = []
    for name, transformer, cols in ct.transformers_:
        if name == "cat":
            names.extend(transformer.get_feature_names_out(cols))
        elif name == "num":
            names.extend(cols)
    return np.array(names)

# ══════════════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION (5-fold)
# ══════════════════════════════════════════════════════════════════════
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run_cv(model_dict):
    rows = []
    for y_col, y_label in TARGETS:
        y = df[y_col]
        for spec_label, feat_cols in model_dict.items():
            X = df[feat_cols]
            res = cross_validate(lr_pipe(feat_cols), X, y,
                                 cv=cv, scoring=["roc_auc", "accuracy"])
            rows.append({
                "target":   y_label,
                "model":    spec_label.replace("\n", " "),
                "auc_mean": res["test_roc_auc"].mean(),
                "auc_std":  res["test_roc_auc"].std(),
                "acc_mean": res["test_accuracy"].mean(),
                "acc_std":  res["test_accuracy"].std(),
            })
    return pd.DataFrame(rows)

print("\nRunning 5-fold CV — incremental models …")
incr_df = run_cv(INCR_MODELS)
print(incr_df.to_string(index=False))

print("\nRunning 5-fold CV — ablation models …")
abla_df = run_cv(ABLA_MODELS)
print(abla_df.to_string(index=False))

incr_df.to_csv(os.path.join(OUT_TABLES, "model_performance_incremental.csv"), index=False)
abla_df.to_csv(os.path.join(OUT_TABLES, "model_performance_ablation.csv"),    index=False)

# ══════════════════════════════════════════════════════════════════════
# PLOT 1 — Incremental model comparison (AUC)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(INCR_MODELS))

for ax, (y_col, y_label) in zip(axes, TARGETS):
    sub = incr_df[incr_df["target"] == y_label].reset_index(drop=True)
    bars = ax.bar(x, sub["auc_mean"], width=0.55, color=PALETTE,
                  yerr=sub["auc_std"], capsize=5,
                  error_kw={"linewidth": 1.5, "ecolor": "#555"})
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(INCR_MODELS.keys()), fontsize=9.5)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("ROC-AUC (5-fold CV)")
    ax.set_title(f"Predicting {y_label}", fontweight="bold")
    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["auc_std"] + 0.013,
                f'{row["auc_mean"]:.3f}', ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=9)

plt.suptitle("Logistic Regression — Incremental Model Comparison\n"
             "(votes_received excluded)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "01_model_comparison_incremental.png"), dpi=150)
plt.close()
print("\nSaved: 01_model_comparison_incremental.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 2 — Ablation: standalone contribution
# ══════════════════════════════════════════════════════════════════════
abla_colors = ["#cccccc", "#a9a7c7", "#e8cc7a", "#e48375", "#7fb8b0"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(ABLA_MODELS))

for ax, (y_col, y_label) in zip(axes, TARGETS):
    sub = abla_df[abla_df["target"] == y_label].reset_index(drop=True)
    bars = ax.bar(x, sub["auc_mean"], width=0.55, color=abla_colors,
                  yerr=sub["auc_std"], capsize=5,
                  error_kw={"linewidth": 1.5, "ecolor": "#555"})
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    bars[-1].set_edgecolor("black")
    bars[-1].set_linewidth(1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(list(ABLA_MODELS.keys()), fontsize=9.5)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("ROC-AUC (5-fold CV)")
    ax.set_title(f"Predicting {y_label}", fontweight="bold")
    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["auc_std"] + 0.013,
                f'{row["auc_mean"]:.3f}', ha="center", va="bottom", fontsize=9)

plt.suptitle("Ablation Study — Standalone Feature Group Contribution\n"
             "(votes_received excluded)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "02_ablation_standalone.png"), dpi=150)
plt.close()
print("Saved: 02_ablation_standalone.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 3 — ROC curves (all 4 incremental models)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (y_col, y_label) in zip(axes, TARGETS):
    y = df[y_col]
    for (spec_label, feat_cols), color in zip(INCR_MODELS.items(), PALETTE):
        X = df[feat_cols]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        pipe = lr_pipe(feat_cols)
        pipe.fit(X_tr, y_tr)
        fpr, tpr, _ = roc_curve(y_te, pipe.predict_proba(X_te)[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{spec_label.replace(chr(10), " ")}  (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {y_label}", fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")

plt.suptitle("Logistic Regression ROC Curves  (votes_received excluded, 80/20 split)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "03_roc_curves.png"), dpi=150)
plt.close()
print("Saved: 03_roc_curves.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 4 & 5 — Coefficient plots (Model 4)
# ══════════════════════════════════════════════════════════════════════
for y_col, y_label in TARGETS:
    pipe = lr_pipe(ALL_COLS)
    pipe.fit(df[ALL_COLS], df[y_col])

    feat_names = get_feature_names(pipe, ALL_COLS)
    coef_df    = (pd.DataFrame({"feature": feat_names,
                                "coef": pipe.named_steps["clf"].coef_[0]})
                  .assign(abs_coef=lambda d: d["coef"].abs())
                  .sort_values("abs_coef"))

    colors = ["#e48375" if c > 0 else "#7fb8b0" for c in coef_df["coef"]]

    fig, ax = plt.subplots(figsize=(9, max(6, len(coef_df) * 0.35)))
    ax.barh(coef_df["feature"], coef_df["coef"], color=colors, height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient (L2-regularized)")
    ax.set_title(f"Model 4 Coefficients — {y_label}\n(votes_received excluded)",
                 fontweight="bold")
    pos_patch = mpatches.Patch(color="#e48375", label="Positive → increases probability")
    neg_patch = mpatches.Patch(color="#7fb8b0", label="Negative → decreases probability")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right")
    plt.tight_layout()
    fname = f"04_coefficients_{y_col}.png"
    plt.savefig(os.path.join(OUT_PLOTS, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

    coef_df.drop(columns="abs_coef").sort_values("coef", key=abs, ascending=False).to_csv(
        os.path.join(OUT_TABLES, f"coefficients_{y_col}.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════
# PLOT 6 — Decision Tree (survival, depth = 3)
# ══════════════════════════════════════════════════════════════════════
print("\nFitting Decision Tree …")
X = df[ALL_COLS]
y = df["survive"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

dt = dt_pipe(ALL_COLS, max_depth=3)
dt.fit(X_tr, y_tr)
dt_acc = accuracy_score(y_te, dt.predict(X_te))
print(f"Decision Tree test accuracy: {dt_acc:.3f}")

feat_names_dt = get_feature_names(dt, ALL_COLS)
fig, ax = plt.subplots(figsize=(22, 8))
plot_tree(dt.named_steps["clf"],
          feature_names=feat_names_dt,
          class_names=["Eliminated", "Survived"],
          filled=True, rounded=True, fontsize=9,
          impurity=False, proportion=True, ax=ax)
ax.set_title(f"Decision Tree — Player Survival  (depth=3,  test acc={dt_acc:.3f})\n"
             "votes_received excluded",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "06_decision_tree_survival.png"),
            dpi=120, bbox_inches="tight")
plt.close()
print("Saved: 06_decision_tree_survival.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 7 — Decision Tree feature importance
# ══════════════════════════════════════════════════════════════════════
imp_df = (pd.DataFrame({"feature": feat_names_dt,
                        "importance": dt.named_steps["clf"].feature_importances_})
          .query("importance > 0")
          .sort_values("importance"))

fig, ax = plt.subplots(figsize=(8, max(4, len(imp_df) * 0.42)))
ax.barh(imp_df["feature"], imp_df["importance"], color="#a9a7c7", height=0.65)
ax.set_xlabel("Gini Importance")
ax.set_title("Decision Tree Feature Importance — Player Survival\n"
             "(votes_received excluded)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "07_dt_feature_importance.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_dt_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 8 & 9 — Per-role and per-LLM accuracy (Model 4)
# ══════════════════════════════════════════════════════════════════════
pipe_full = lr_pipe(ALL_COLS)
pipe_full.fit(df[ALL_COLS], df["survive"])
df["pred_survive"] = pipe_full.predict(df[ALL_COLS])

role_acc = (df.groupby("role")
              .apply(lambda g: accuracy_score(g["survive"], g["pred_survive"]),
                     include_groups=False)
              .reset_index(name="accuracy")
              .sort_values("accuracy", ascending=False))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(role_acc["role"],
       role_acc["accuracy"],
       color=[ROLE_COLORS.get(r, "#ccc") for r in role_acc["role"]],
       width=0.5)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Survival Prediction Accuracy by Role  (Model 4, votes_received excluded)",
             fontweight="bold")
for i, row in enumerate(role_acc.itertuples()):
    ax.text(i, row.accuracy + 0.012, f"{row.accuracy:.3f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "08_accuracy_by_role.png"), dpi=150)
plt.close()
print("Saved: 08_accuracy_by_role.png")

model_acc = (df.groupby("model_name")
               .apply(lambda g: accuracy_score(g["survive"], g["pred_survive"]),
                      include_groups=False)
               .reset_index(name="accuracy")
               .sort_values("accuracy", ascending=False))

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(range(len(model_acc)), model_acc["accuracy"], color="#7fb8b0", width=0.6)
ax.set_xticks(range(len(model_acc)))
ax.set_xticklabels(model_acc["model_name"], rotation=20, ha="right", fontsize=9.5)
ax.set_ylim(0, 1)
ax.set_ylabel("Survival Prediction Accuracy")
ax.set_title("Survival Prediction Accuracy by LLM  (Model 4, votes_received excluded)",
             fontweight="bold")
for i, row in enumerate(model_acc.itertuples()):
    ax.text(i, row.accuracy + 0.01, f"{row.accuracy:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "09_accuracy_by_llm.png"), dpi=150)
plt.close()
print("Saved: 09_accuracy_by_llm.png")

# ── Done ───────────────────────────────────────────────────────────────
print(f"\nAll outputs → {os.path.relpath(os.path.join(HERE, 'outputs_'))}/")
print("  plots/  : 09 PNG files")
print("  tables/ : model_performance_incremental.csv, model_performance_ablation.csv,")
print("            coefficients_survive.csv, coefficients_player_win.csv")
