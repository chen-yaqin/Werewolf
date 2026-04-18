"""Public-speech feature analysis and visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.common import add_quantile_bin, build_player_feature_table, load_tables, prepare_output_dirs, setup_plot_style


def run(processed_root: str | Path, output_root: str | Path) -> None:
    setup_plot_style()
    tables = load_tables(processed_root)
    players = tables["players"]
    votes = tables["votes"]
    speeches = tables["speeches"]
    night_actions = tables["night_actions"]

    feature_table = build_player_feature_table(players, votes, speeches, night_actions)
    tables_dir, figures_dir = prepare_output_dirs(output_root, "03_speech_analysis")

    speech_summary = (
        feature_table.groupby("role", observed=False, as_index=False)
        .agg(
            player_games=("player_id", "size"),
            mean_public_messages=("public_message_count", "mean"),
            mean_public_words=("public_word_count", "mean"),
            mean_avg_words_per_message=("avg_words_per_message", "mean"),
            mean_question_rate=("question_rate", "mean"),
            win_rate=("won", "mean"),
        )
        .sort_values("role")
    )
    speech_summary.to_csv(tables_dir / "speech_summary_by_role.csv", index=False)

    quartile_df = add_quantile_bin(feature_table.copy(), "public_message_count", "message_count_bin")
    win_rate_by_bin = (
        quartile_df.groupby("message_count_bin", as_index=False)
        .agg(
            player_games=("player_id", "size"),
            mean_messages=("public_message_count", "mean"),
            mean_words=("public_word_count", "mean"),
            win_rate=("won", "mean"),
        )
        .sort_values("mean_messages")
    )
    win_rate_by_bin.to_csv(tables_dir / "win_rate_by_message_bin.csv", index=False)

    if not feature_table.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=feature_table, x="role", y="public_message_count", ax=ax, color="#D69C4E")
        ax.set_title("Public Message Count by Role")
        ax.set_xlabel("Role")
        ax.set_ylabel("Public messages per player-game")
        fig.savefig(figures_dir / "public_message_count_by_role.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=feature_table, x="role", y="avg_words_per_message", ax=ax, color="#8A9A5B")
        ax.set_title("Average Message Length by Role")
        ax.set_xlabel("Role")
        ax.set_ylabel("Average words per public message")
        fig.savefig(figures_dir / "avg_words_per_message_by_role.png")
        plt.close(fig)

    if not win_rate_by_bin.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=win_rate_by_bin, x="message_count_bin", y="win_rate", ax=ax, color="#A55C4A")
        ax.set_title("Win Rate by Public Message Volume")
        ax.set_xlabel("Public message count bin")
        ax.set_ylabel("Win rate")
        ax.tick_params(axis="x", rotation=25)
        fig.savefig(figures_dir / "win_rate_by_message_volume.png")
        plt.close(fig)

    if not speeches.empty:
        scatter_df = feature_table[["public_message_count", "public_word_count", "won", "role"]].copy()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=scatter_df,
            x="public_message_count",
            y="public_word_count",
            hue="won",
            style="role",
            alpha=0.5,
            ax=ax,
        )
        ax.set_title("Speech Volume vs Word Count")
        ax.set_xlabel("Public message count")
        ax.set_ylabel("Public word count")
        fig.savefig(figures_dir / "speech_volume_scatter.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run(processed_root=args.processed_root, output_root=args.output_root)


if __name__ == "__main__":
    main()
