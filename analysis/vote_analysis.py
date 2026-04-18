"""Voting-feature analysis and visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.common import build_player_feature_table, load_tables, prepare_output_dirs, setup_plot_style


def run(processed_root: str | Path, output_root: str | Path) -> None:
    setup_plot_style()
    tables = load_tables(processed_root)
    players = tables["players"]
    votes = tables["votes"]
    speeches = tables["speeches"]
    night_actions = tables["night_actions"]

    feature_table = build_player_feature_table(players, votes, speeches, night_actions)
    tables_dir, figures_dir = prepare_output_dirs(output_root, "02_vote_analysis")

    day_votes = votes[votes["is_day_vote"] == 1].copy() if not votes.empty else pd.DataFrame()

    vote_summary = (
        feature_table.groupby("role", observed=False, as_index=False)
        .agg(
            player_games=("player_id", "size"),
            mean_vote_accuracy=("vote_accuracy", "mean"),
            mean_votes_received=("day_votes_received", "mean"),
            mean_majority_alignment=("vote_majority_alignment", "mean"),
        )
        .sort_values("role")
    )
    vote_summary.to_csv(tables_dir / "vote_summary_by_role.csv", index=False)

    win_rate_by_day1 = (
        feature_table.groupby("day1_voted_wolf", as_index=False)
        .agg(
            player_games=("player_id", "size"),
            win_rate=("won", "mean"),
            mean_vote_accuracy=("vote_accuracy", "mean"),
        )
        .sort_values("day1_voted_wolf")
    )
    win_rate_by_day1.to_csv(tables_dir / "win_rate_by_day1_voted_wolf.csv", index=False)

    if not day_votes.empty:
        day1_votes = day_votes[day_votes["day"] == 1].copy()
        if not day1_votes.empty:
            heatmap = pd.crosstab(day1_votes["actor_role"], day1_votes["target_role"])
            heatmap.to_csv(tables_dir / "day1_vote_target_heatmap.csv")

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(heatmap, annot=True, fmt="d", cmap="YlOrBr", ax=ax)
            ax.set_title("Day 1 Vote Targets by Actor Role")
            ax.set_xlabel("Target role")
            ax.set_ylabel("Actor role")
            fig.savefig(figures_dir / "day1_vote_target_heatmap.png")
            plt.close(fig)

    if not win_rate_by_day1.empty:
        label_map = {0: "No", 1: "Yes"}
        plot_df = win_rate_by_day1.copy()
        plot_df["day1_voted_wolf"] = plot_df["day1_voted_wolf"].map(label_map).fillna("Unknown")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=plot_df, x="day1_voted_wolf", y="win_rate", ax=ax, color="#5E8C61")
        ax.set_title("Winning Rate vs Day 1 Vote Correctness")
        ax.set_xlabel("Voted for a werewolf on Day 1")
        ax.set_ylabel("Win rate")
        fig.savefig(figures_dir / "win_rate_by_day1_vote_correctness.png")
        plt.close(fig)

    if not feature_table.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=feature_table, x="won", y="day_votes_received", ax=ax, color="#9D6B53")
        ax.set_title("Votes Received vs Final Outcome")
        ax.set_xlabel("Won")
        ax.set_ylabel("Day votes received")
        fig.savefig(figures_dir / "day_votes_received_by_outcome.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=vote_summary, x="role", y="mean_vote_accuracy", ax=ax, color="#5677A4")
        ax.set_title("Mean Vote Accuracy by Role")
        ax.set_xlabel("Role")
        ax.set_ylabel("Average fraction of day votes on werewolves")
        fig.savefig(figures_dir / "mean_vote_accuracy_by_role.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run(processed_root=args.processed_root, output_root=args.output_root)


if __name__ == "__main__":
    main()
