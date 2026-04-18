# Werewolf Project

本项目已经完成数据清洗、并行抽取与结果合并。下一阶段正式进入 `analysis/`，基于 merged Parquet 做描述统计、行为分析与建模。

## 当前进度

在正式分析之前，我们已经完成了整套数据预处理与并行抽取流程：

1. 从约 `57G` 的原始候选数据中抽取出约 `12G` 的 `data.tar`，上传到服务器并解压为 `data/` 目录。
2. 初步检查发现共有 `5845` 个 JSON 文件，其中 `4405` 个是 `0` 字节空文件，另有 `5` 个非空但损坏的 JSON 文件。
3. 清洗后最终保留 `1435` 个有效游戏日志，作为后续全部分析的数据基础。
4. 在服务器上配置了项目虚拟环境，安装依赖并设置 `PYTHONPATH`，同时修复了 Slurm 日志路径、partition 不可用、环境缺失等问题，保证批处理流程可稳定运行。
5. 将 `1435` 个有效 JSON 按每 `200` 个文件切分为 `8` 个 chunk，并使用增强版 `scripts/process_chunk.py` 提取四类结构化信息：
   - game-level
   - player-level
   - public message-level
   - event-level
6. 使用 Slurm array jobs（task `0-7`）并行处理 `8` 个 chunk，成功生成对应的 chunk-level 输出。
7. 最终将所有 chunk 结果合并为 `5` 张 Parquet 表：
   - `games.parquet`: `1435` 行
   - `players.parquet`: `11472` 行
   - `public_messages.parquet`: `40510` 行
   - `events.parquet`: `407262` 行
   - `errors.parquet`: 空表

到这里，数据预处理与并行抽取阶段已经完成。后续所有描述性统计、可视化与建模分析都直接基于这些 Parquet 表进行，不需要再回到原始 JSON，也不需要再次并行。课程要求中的 parallel computing 部分，已经在 chunk 处理阶段完成。

## 目录说明

当前仓库的核心目录如下：

```text
Werewolf/
├─ analysis/
│  ├─ descriptive_analysis/
│  ├─ vote_analysis/
│  ├─ speech_analysis/
│  ├─ role_analysis/
│  └─ regression_models/
├─ scripts/
│  ├─ make_chunks.py
│  ├─ process_chunk.py
│  └─ merge_outputs.py
├─ slurm/
│  ├─ 00_create_env.sbatch
│  ├─ 01_extract_data.sbatch
│  ├─ 02_make_chunks.sbatch
│  ├─ 03_process_chunks_array.sbatch
│  └─ 04_merge_outputs.sbatch
└─ README.md
```

## `download/` 里有什么

在当前 workspace 中，预处理产物位于仓库同级目录的 `download/` 下。关键内容如下：

```text
download/
└─ download/
   ├─ data_cleaned.tar
   ├─ outputs.tar
   └─ outputs/
      └─ outputs/
         ├─ chunk_results/
         └─ merged/
```

各部分含义：

- `download/download/data_cleaned.tar`

  - 清洗后的有效 JSON 数据打包结果。
  - 对应最终保留下来的 `1435` 个有效游戏日志。
- `download/download/outputs.tar`

  - 并行抽取产物的整体打包文件。
  - 便于迁移、备份或重新分发输出结果。
- `download/download/outputs/outputs/chunk_results/`

  - `8` 个 chunk 的逐块输出结果。
  - 每个 chunk 都会生成一组 CSV，例如：
    - `games_chunk_00000.csv`
    - `players_chunk_00000.csv`
    - `public_messages_chunk_00000.csv`
    - `events_chunk_00000.csv`
    - `errors_chunk_00000.csv`
  - 这部分主要用于检查 chunk 级别是否正常，以及必要时回溯具体块的处理结果。
- `download/download/outputs/outputs/merged/`

  - 所有 chunk 合并后的最终分析数据。
  - 这是后续 `analysis/` 应该直接读取的目录。
  - 当前包含：
    - `games.parquet`
    - `players.parquet`
    - `public_messages.parquet`
    - `events.parquet`
    - `errors.parquet`

## Final Parquet 结构

后续分析默认读取 `download/download/outputs/outputs/merged/` 中的 5 张表。

### `games.parquet`

每局游戏一行，主要字段：

- `game_id`
- `filename`
- `winner_team`
- `last_day`
- `n_players`
- `end_reason`

用途：做 game-level 描述统计，例如总局数、胜负分布、游戏长度分布。

### `players.parquet`

每局中每位玩家一行，主要字段：

- `game_id`
- `player_id`
- `role`
- `model_name`
- `alive_end`
- `eliminated_during_day`
- `eliminated_during_phase`

用途：做 player-level 统计，例如角色分布、生存率、不同角色的白天/夜晚淘汰情况。

### `public_messages.parquet`

每条公开发言一行，主要字段：

- `game_id`
- `filename`
- `day`
- `phase`
- `speaker_id`
- `event_name`
- `text`
- `text_len`
- `created_at`

用途：做公开发言行为分析，例如发言次数、发言长度、首日发言活跃度。

### `events.parquet`

每条事件一行，是最完整的行为流水表，主要字段：

- `game_id`
- `filename`
- `outer_idx`
- `inner_idx`
- `data_type`
- `event_name`
- `day`
- `phase`
- `detailed_phase`
- `source`
- `public`
- `visible_in_ui`
- `created_at`
- `actor_id`
- `target_id`
- `reasoning`
- `description`

用途：做投票分析、角色行动分析，以及必要时补查上下文。

### `errors.parquet`

异常记录表，字段：

- `filepath`
- `error`

当前为空，说明 merged 数据中没有残留解析错误。

## 下一步：正式进入 `analysis/`

接下来所有工作围绕下面 5 个子目录展开：

```text
analysis/
├─ descriptive_analysis/
├─ vote_analysis/
├─ speech_analysis/
├─ role_analysis/
└─ regression_models/
```

推荐按“先总览、再三条主线、最后整合模型”的顺序推进。

## 每个分析文件夹要做什么

### `analysis/descriptive_analysis`

目标：先把数据本身讲清楚，作为 report 和 slides 的开头。

需要分析：

- 总游戏数
- `winner_team` 分布
- `last_day` 分布
- 玩家角色分布
- 各角色 survival rate
- 每局公开发言数分布
- 每条消息平均长度
- 事件类型频次

建议代码：

- `01_overview_stats.py`
  - 读取 `games.parquet`、`players.parquet`、`public_messages.parquet`、`events.parquet`
  - 输出 summary text 和 summary csv
- `02_overview_plots.py`
  - 生成基础描述图

建议输出：

- `winner_team_counts.csv`
- `role_counts.csv`
- `role_survival_rates.csv`
- `game_length_counts.csv`
- `messages_per_game_summary.csv`

建议图：

- Winner team bar chart
- Game length histogram / bar chart
- Role count bar chart
- Role survival rate bar chart
- Messages per game histogram

### `analysis/vote_analysis`

目标：分析投票模式与胜负、出局之间的关系。

需要分析：

- 白天投票事件与夜间狼人投票事件
- 玩家投票次数、被投票次数
- 是否容易出现分票或 tie
- 投票集中度是否和胜负相关
- 狼人夜间 targeting 是否更一致

建议代码：

- `01_extract_vote_features.py`
  - 从 `events.parquet` 中提取 vote-related rows
  - 构建 player-level 和 game-level 投票特征
- `02_vote_summary.py`
  - 输出描述统计表
- `03_vote_plots.py`
  - 生成投票相关图

建议输出：

- `vote_events.parquet` 或 `vote_events.csv`
- `vote_features_by_player.csv`
- `vote_features_by_game.csv`

建议图：

- Votes received by outcome
- Vote concentration by winning team
- Night vote agreement rate for werewolves
- Representative vote flow chart（可选）

### `analysis/speech_analysis`

目标：基于公开发言做轻量、可解释的行为分析，不做复杂 NLP。

需要分析：

- 每位玩家的发言次数
- 每位玩家的总发言长度与平均发言长度
- 首日发言活跃度
- 玩家发言活跃度与生存/胜负关系
- 不同角色的说话模式差异

建议代码：

- `01_extract_speech_features.py`
  - 按 `game_id + speaker_id` 聚合公开发言
- `02_speech_summary.py`
  - 输出描述统计
- `03_speech_plots.py`
  - 生成发言相关图

建议输出：

- `speech_features_by_player.csv`
- `speech_features_by_game.csv`

建议图：

- Message count by role
- Message count by win/loss
- Average message length by role
- First-day message count vs survival / win
- Top speakers per game distribution（可选）

### `analysis/role_analysis`

目标：分析 Werewolf 中最有特色的角色机制，尤其是夜间行动。

需要分析：

- Seer 的 inspect 行为
- Doctor 的 heal 行为
- Werewolf 的夜间 targeting 行为
- 各角色白天放逐与夜晚淘汰比例
- 各角色活到最后的比例

建议代码：

- `01_extract_role_action_features.py`
  - 从 `events.parquet` 中提取角色行动相关事件
- `02_role_summary.py`
  - 生成角色行动统计表
- `03_role_plots.py`
  - 生成角色相关图

建议输出：

- `seer_features.csv`
- `doctor_features.csv`
- `werewolf_night_features.csv`
- 或合并后的 `role_action_features.csv`

建议图：

- Role survival rate
- Role elimination phase distribution
- Seer found wolf vs villager win rate
- First-night target role distribution
- Doctor action frequency / inferred save success（可选）

### `analysis/regression_models`

目标：把投票、发言和角色行动三条线整合成统一模型，作为最后的总结部分。

需要分析：

- 玩家所在阵营是否获胜
- 玩家是否活到最后
- 哪类特征最有解释力
- 加入投票、发言、角色行动后模型是否提升

建议代码：

- `01_build_model_table.py`
  - 合并 vote、speech、role 三条线特征，得到 player-level modeling table
- `02_logistic_regression.py`
  - 建立 logistic regression
- `03_decision_tree.py`
  - 可选，作为补充模型

建议模型版本：

- Model 1: 基础特征（`role`、`model_name`）
- Model 2: 基础 + 投票特征
- Model 3: 基础 + 投票 + 发言特征
- Model 4: 基础 + 投票 + 发言 + 角色行动特征

建议图：

- Logistic regression coefficient plot
- Model performance comparison
- Decision tree plot（可选）

## 推荐推进顺序

建议按下面顺序开展：

1. `descriptive_analysis`
2. `vote_analysis`
3. `speech_analysis`
4. `role_analysis`
5. `regression_models`

原因很简单：

- `descriptive_analysis` 能先确认 merged Parquet 是否正常，并直接产出 slides 开头要用的图。
- `vote_analysis` 依赖 `events.parquet` 最多，应该尽早把事件筛选逻辑跑通。
- `speech_analysis` 和 `role_analysis` 可以在前两部分稳定后并行推进。
- `regression_models` 需要前面三条特征线都准备好，适合最后做整合。

## 团队协作建议

为了减少冲突，建议大家先认领自己的任务，再分别提交自己负责的代码。

推荐方式：

1. 每个人认领一个分析子目录，或认领明确的脚本文件。
2. 尽量不要多人同时改同一个脚本。
3. 每个人只 push 自己负责部分的代码。
4. 如果需要改公共文件，先在组内说明，避免覆盖别人修改。

一个简单可执行的分工方式是：

- A 负责 `descriptive_analysis`
- B 负责 `vote_analysis`
- C 负责 `speech_analysis`
- D 负责 `role_analysis`
- E 负责 `regression_models`
- 最后汇总结论+ppt+report

## 总结

我们已经完成了从原始 JSON 到 merged Parquet 的清洗、并行抽取与合并；接下来团队只需要围绕 `analysis/` 五个子目录认领任务、编写分析脚本、生成图表并完成最终报告即可。
