# Werewolf Game Analysis

This repository contains the completed code, analysis outputs, and presentation-ready materials for a large-scale study of AI agents playing the social deduction game Werewolf. The project converts raw game logs into structured tables, extracts behavioral features, analyzes role-specific and communication patterns, trains predictive models, and provides an interactive Streamlit dashboard for exploring games and player behavior.

The final project deliverables include the analysis code, generated tables and figures, the written report, and the presentation slides.

## Project Overview

Werewolf is a hidden-role game in which villagers try to identify and eliminate werewolves while werewolves coordinate secretly to remove villagers. Because the game depends on speech, voting, deception, role actions, and coalition behavior, it provides a useful environment for studying multi-agent LLM behavior.

This project studies 1,435 valid Werewolf games played by LLM agents. The analysis focuses on four main questions:

1. How do outcomes, role distributions, game length, and survival rates vary across games?
2. How are voting behavior, vote concentration, and tie frequency related to survival and team victory?
3. How do public speech patterns differ by role, survival status, and winning team?
4. Which behavioral features best predict player survival and whether a player's team wins?

## Dataset

The raw data began as a large collection of candidate JSON game logs. The preprocessing pipeline identified valid logs, removed empty or corrupted files, and extracted structured information into analysis-ready tables.

Summary of the processed dataset:

| Item                                   |   Count |
| -------------------------------------- | ------: |
| Candidate JSON files                   |   5,845 |
| Empty JSON files removed               |   4,405 |
| Corrupted non-empty JSON files removed |       5 |
| Valid games retained                   |   1,435 |
| Player-game rows                       |  11,472 |
| Public messages                        |  40,510 |
| Event records                          | 407,262 |
| Parsing errors in merged output        |       0 |

The final merged tables are:

| Table                       | Unit of observation          | Description                                                |
| --------------------------- | ---------------------------- | ---------------------------------------------------------- |
| `games.parquet`           | One row per game             | Game outcome, game length, player count, and end reason    |
| `players.parquet`         | One row per player per game  | Role, model name, survival status, and elimination timing  |
| `public_messages.parquet` | One row per public message   | Speaker, day, phase, message text, and message length      |
| `events.parquet`          | One row per structured event | Full event stream used for voting and role-action analysis |
| `errors.parquet`          | One row per failed parse     | Empty in the final processed output                        |

Large raw and intermediate data archives are not tracked directly in this repository. The analysis scripts expect merged data to be available in the project data directory used during preprocessing, while the generated analysis outputs are included under `analysis/`.

## Repository Structure

```text
Werewolf/
├─ analysis/
│  ├─ descriptive_analysis/      # Overview statistics and baseline plots
│  ├─ vote_analysis/             # Vote-event extraction, vote features, and plots
│  ├─ speech_analysis/           # Public-message features and speech plots
│  ├─ role_analysis/             # Seer, Doctor, and Werewolf role-action features
│  ├─ regression_models/         # Predictive modeling outputs and figures
│  └─ visualization/             # Interactive Streamlit dashboard
├─ scripts/
│  ├─ make_chunks.py             # Build chunk manifests for parallel processing
│  ├─ process_chunk.py           # Extract structured CSV rows from JSON logs
│  └─ merge_outputs.py           # Merge chunk-level CSVs into final tables
├─ slurm/
│  ├─ 00_create_env.sbatch
│  ├─ 01_extract_data.sbatch
│  ├─ 02_make_chunks.sbatch
│  ├─ 03_process_chunks_array.sbatch
│  └─ 04_merge_outputs.sbatch
├─ requirements.txt
└─ README.md
```

## Methodology

### 1. Parallel Data Processing

The raw logs were cleaned and processed using a chunk-based workflow designed for Slurm:

1. Valid JSON files were identified after removing empty and corrupted logs.
2. The 1,435 valid logs were split into 8 chunk manifests.
3. Slurm array jobs processed chunks in parallel.
4. Each chunk produced CSV files for games, players, public messages, events, and errors.
5. Chunk outputs were merged into final Parquet tables for analysis.

The preprocessing stage produced the structured data foundation for all later analysis and satisfies the parallel-computing component of the project.

### 2. Descriptive Analysis

The descriptive analysis summarizes the full dataset and creates overview plots for:

- Winner-team distribution
- Game length distribution
- Role counts
- Role survival rates
- Public messages per game
- Event-type frequencies

Key descriptive results:

- Villagers won 1,000 of 1,435 games, or 69.7%.
- Games contained an average of 7.99 players.
- Public messages averaged 28.23 messages per game.
- The average public message length was 548.32 characters.
- The most frequent event type was `phase_change`, with 103,287 records.
- Villagers had the highest role-level survival rate at 54.8%.

### 3. Voting Analysis

The voting analysis extracts `vote_action` events and separates day votes from night votes. It creates player-level and game-level features such as:

- Votes cast
- Votes received
- Day votes cast and received
- Night votes cast and received
- Vote concentration
- Tie frequency
- Average night-vote agreement

Generated outputs are stored in `analysis/vote_analysis/outputs/`, with figures in `analysis/vote_analysis/plots/`.

### 4. Speech Analysis

The speech analysis aggregates public messages by player and game. It measures:

- Number of messages
- Total text length
- Average message length
- First-day message count
- First-day text length
- Relationships among speech, role, survival, and winning team

Generated tables are stored in `analysis/speech_analysis/Outputs/tables/`, with figures in `analysis/speech_analysis/Outputs/plots/`.

### 5. Role Analysis

The role analysis focuses on mechanics unique to Werewolf:

- Seer inspection behavior
- Doctor healing behavior
- Werewolf night-targeting behavior
- Role-specific survival and elimination patterns

Generated outputs are stored in `analysis/role_analysis/outputs/`, with figures in `analysis/role_analysis/plots/`.

### 6. Predictive Modeling

The modeling stage combines role, model identity, voting, speech, and role-action features. It evaluates two prediction targets:

- Whether a player survives to the end of the game
- Whether a player's team wins

Models include logistic regression for the main analysis and decision trees for interpretable supplementary analysis. Five-fold cross-validation is used to compare incremental feature sets and ablation models.

Best observed cross-validated performance in the generated outputs:

| Target            | Best ROC-AUC | Best Accuracy |
| ----------------- | -----------: | ------------: |
| Player survival   |        0.888 |         0.821 |
| Player's team win |        0.801 |         0.732 |

Model tables and figures are stored in `analysis/regression_models/outputs_/`.

## Interactive Dashboard

The Streamlit dashboard in `analysis/visualization/app.py` provides an interactive way to inspect the final analysis outputs.

Features include:

- Game browser with random-game selection
- Player roster by role, model, and survival status
- Day and night vote timelines
- Public message display when message text is available
- Elimination order
- Player statistics by role, model, survival status, and winning team
- Group comparisons across behavioral metrics

Run the dashboard with:

```powershell
streamlit run analysis/visualization/app.py
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the Data Pipeline

The full preprocessing workflow can be run manually or through the Slurm scripts in `slurm/`.

Create chunk manifests:

```powershell
python scripts/make_chunks.py --data-dir <clean_json_dir> --chunks-dir <chunks_dir> --files-per-chunk 200
```

Process one chunk:

```powershell
python scripts/process_chunk.py --chunk_file <chunks_dir>/chunk_00000.txt --chunk_id 00000 --output_dir <chunk_output_dir>
```

Merge chunk outputs:

```powershell
python scripts/merge_outputs.py --chunks-root <chunk_output_dir> --merged-root <merged_output_dir> --write-format parquet
```

For the original cluster workflow, use the Slurm job scripts in order:

```bash
sbatch slurm/00_create_env.sbatch
sbatch slurm/01_extract_data.sbatch
sbatch slurm/02_make_chunks.sbatch
sbatch slurm/03_process_chunks_array.sbatch
sbatch slurm/04_merge_outputs.sbatch
```

## Reproducing the Analyses

The generated outputs are already included in the repository. To rerun selected analysis components, use the relevant scripts or notebooks under `analysis/`.

Examples:

```powershell
python analysis/descriptive_analysis/01_overview_stats.py
python analysis/descriptive_analysis/02_overview_plots.py
python analysis/regression_models/modeling.py
```

Some analysis scripts and notebooks were executed in the original project environment after the merged data tables were produced. If running them on a new machine, confirm that the local data paths match the expected merged-table location.

## Main Outputs

Important generated artifacts include:

| Location                                   | Contents                                      |
| ------------------------------------------ | --------------------------------------------- |
| `analysis/descriptive_analysis/outputs/` | Overview tables and descriptive plots         |
| `analysis/vote_analysis/outputs/`        | Clean voting events and vote-derived features |
| `analysis/vote_analysis/plots/`          | Voting behavior figures                       |
| `analysis/speech_analysis/Outputs/`      | Speech feature tables and plots               |
| `analysis/role_analysis/outputs/`        | Role-action feature tables                    |
| `analysis/role_analysis/plots/`          | Role-action and survival figures              |
| `analysis/regression_models/outputs_/`   | Model performance tables and model figures    |
| `analysis/visualization/app.py`          | Interactive dashboard                         |
