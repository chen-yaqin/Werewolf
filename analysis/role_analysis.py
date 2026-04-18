"""Role-specific night-action analysis and visualizations."""

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
    tables_dir, figures_dir = prepare_output_dirs(output_root, "04_role_analysis")

    seer_df = feature_table[feature_table["role"] == "Seer"].copy()
    doctor_df = feature_table[feature_table["role"] == "Doctor"].copy()
    wolf_df = feature_table[feature_table["role"] == "Werewolf"].copy()

    seer_summary = pd.DataFrame()
    if not seer_df.empty:
        seer_df["found_any_wolf"] = (seer_df["seer_hit_wolf_count"] > 0).astype(int)
        seer_summary = (
            seer_df.groupby("found_any_wolf", as_index=False)
            .agg(
                player_games=("player_id", "size"),
                win_rate=("won", "mean"),
                mean_inspects=("seer_inspects", "mean"),
                mean_hit_rate=("seer_hit_rate", "mean"),
            )
            .sort_values("found_any_wolf")
        )
        seer_summary.to_csv(tables_dir / "seer_summary.csv", index=False)

    doctor_summary = pd.DataFrame()
    if not doctor_df.empty:
        doctor_df["saved_anyone"] = (doctor_df["doctor_successful_save_count"] > 0).astype(int)
        doctor_summary = (
            doctor_df.groupby("saved_anyone", as_index=False)
            .agg(
                player_games=("player_id", "size"),
                win_rate=("won", "mean"),
                mean_heals=("doctor_heals", "mean"),
                mean_save_rate=("doctor_save_rate", "mean"),
            )
            .sort_values("saved_anyone")
        )
        doctor_summary.to_csv(tables_dir / "doctor_summary.csv", index=False)

    wolf_summary = pd.DataFrame()
    if not wolf_df.empty:
        wolf_df["targeted_power_role"] = (wolf_df["werewolf_targeted_power_role_count"] > 0).astype(int)
        wolf_summary = (
            wolf_df.groupby("targeted_power_role", as_index=False)
            .agg(
                player_games=("player_id", "size"),
                win_rate=("won", "mean"),
                mean_consensus_rate=("werewolf_consensus_rate", "mean"),
                mean_night_votes=("werewolf_night_votes", "mean"),
            )
            .sort_values("targeted_power_role")
        )
        wolf_summary.to_csv(tables_dir / "werewolf_summary.csv", index=False)

    if not seer_summary.empty:
        plot_df = seer_summary.copy()
        plot_df["found_any_wolf"] = plot_df["found_any_wolf"].map({0: "No", 1: "Yes"})
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=plot_df, x="found_any_wolf", y="win_rate", ax=ax, color="#4F7CAC")
        ax.set_title("Seer Win Rate vs Whether a Wolf Was Inspected")
        ax.set_xlabel("Found at least one werewolf")
        ax.set_ylabel("Win rate")
        fig.savefig(figures_dir / "seer_found_wolf_vs_win_rate.png")
        plt.close(fig)

    if not doctor_summary.empty:
        plot_df = doctor_summary.copy()
        plot_df["saved_anyone"] = plot_df["saved_anyone"].map({0: "No", 1: "Yes"})
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=plot_df, x="saved_anyone", y="win_rate", ax=ax, color="#7AA974")
        ax.set_title("Doctor Win Rate vs Successful Save")
        ax.set_xlabel("Successful save at least once")
        ax.set_ylabel("Win rate")
        fig.savefig(figures_dir / "doctor_save_vs_win_rate.png")
        plt.close(fig)

    if not wolf_summary.empty:
        plot_df = wolf_summary.copy()
        plot_df["targeted_power_role"] = plot_df["targeted_power_role"].map({0: "No", 1: "Yes"})
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=plot_df, x="targeted_power_role", y="win_rate", ax=ax, color="#B95D4A")
        ax.set_title("Werewolf Win Rate vs Targeting a Power Role")
        ax.set_xlabel("Targeted a Seer/Doctor at least once")
        ax.set_ylabel("Win rate")
        fig.savefig(figures_dir / "werewolf_target_power_role_vs_win_rate.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run(processed_root=args.processed_root, output_root=args.output_root)


if __name__ == "__main__":
    main()
