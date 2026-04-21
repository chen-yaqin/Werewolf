"""
Werewolf Game Explorer — Interactive Streamlit App
Run: streamlit run analysis/visualization/app.py
"""

import html as _html
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐺 Werewolf Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent.parent   # → analysis/

ROLE_COLORS = {
    "Villager": "#e8cc7a",
    "Werewolf": "#7fb8b0",
    "Seer":     "#a9a7c7",
    "Doctor":   "#e48375",
}
WIN_COLORS = {"Villagers": "#a9a7c7", "Werewolves": "#7fb8b0"}

# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    speech   = pd.read_csv(BASE / "speech_analysis/Outputs/tables/speech_features_by_player.csv")
    vote_p   = pd.read_csv(BASE / "vote_analysis/outputs/vote_features_by_player.csv")
    role_p   = pd.read_csv(BASE / "role_analysis/outputs/role_features_by_player.csv")
    vote_ev  = pd.read_csv(BASE / "vote_analysis/outputs/vote_events_clean.csv")

    _shared = ["role", "model_name", "alive_end", "eliminated_during_day",
               "eliminated_during_phase", "winner_team", "last_day", "n_players", "end_reason"]

    players = (speech
               .merge(vote_p.drop(columns=_shared), on=["game_id", "player_id"])
               .merge(role_p[["game_id", "player_id",
                              "n_inspects", "n_found_wolf", "inspect_success_rate",
                              "n_heals", "n_successful_heals", "heal_success_rate",
                              "n_wolf_votes", "wolf_day_consistency_rate"]],
                      on=["game_id", "player_id"]))
    players = players[players["winner_team"].isin(["Villagers", "Werewolves"])].copy()

    # game-level summary
    games = (players.groupby("game_id")
             .agg(winner_team=("winner_team", "first"),
                  last_day=("last_day", "first"),
                  n_players=("n_players", "first"),
                  n_survived=("alive_end", "sum"),
                  end_reason=("end_reason", "first"))
             .reset_index())

    # optional: public messages (exported from server parquet)
    msg_path = BASE / "speech_analysis/Outputs/tables/public_messages.csv"
    messages = pd.read_csv(msg_path) if msg_path.exists() else None

    return players, vote_ev, games, messages


players, vote_ev, games, messages = load_data()

METRIC_META = {
    "n_messages":           "Message Count",
    "avg_text_len":         "Avg Message Length (chars)",
    "total_text_len":       "Total Characters Written",
    "first_day_messages":   "Day-1 Messages",
    "first_day_text_len":   "Day-1 Total Length",
    "n_votes_received":     "Votes Received",
    "n_votes_cast":         "Votes Cast",
    "n_day_votes_received": "Day Votes Received",
    "n_day_votes_cast":     "Day Votes Cast",
    "n_night_votes_cast":   "Night Votes Cast",
    "n_inspects":           "Seer Inspections",
    "inspect_success_rate": "Seer Success Rate",
    "n_heals":              "Doctor Heals",
    "heal_success_rate":    "Doctor Heal Success Rate",
    "n_wolf_votes":         "Werewolf Night Votes",
}

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — global filters (shared across all tabs)
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🐺 Werewolf Explorer")
    st.markdown("---")
    st.markdown("### Global Filters")

    winner_f = st.multiselect(
        "Winner team", ["Villagers", "Werewolves"],
        default=["Villagers", "Werewolves"])

    role_f = st.multiselect(
        "Role", ["Villager", "Werewolf", "Seer", "Doctor"],
        default=["Villager", "Werewolf", "Seer", "Doctor"])

    model_f = st.multiselect(
        "LLM Model", sorted(players["model_name"].unique()),
        default=sorted(players["model_name"].unique()))

    alive_f = st.multiselect(
        "Survival", ["Survived", "Eliminated"],
        default=["Survived", "Eliminated"])
    alive_vals = [True if a == "Survived" else False for a in alive_f]

    day_range = st.slider(
        "Game length (days)",
        int(games["last_day"].min()), int(games["last_day"].max()),
        (int(games["last_day"].min()), int(games["last_day"].max())))

    st.markdown("---")
    st.caption("Data: 1,435 games · 11,472 players · 8 LLM models")

# Apply global filters
df = players[
    players["winner_team"].isin(winner_f) &
    players["role"].isin(role_f) &
    players["model_name"].isin(model_f) &
    players["alive_end"].isin(alive_vals) &
    players["last_day"].between(*day_range)
].copy()

games_f = games[
    games["winner_team"].isin(winner_f) &
    games["last_day"].between(*day_range)
]

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎮 Game Browser", "📊 Statistics", "🔬 Comparison"])

# ──────────────────────────────────────────────────────────────────────
# TAB 1 — GAME BROWSER
# ──────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🎮 Game Browser")
    st.caption("Select a game to see its full process: player roster, vote timeline, and elimination order.")

    left, right = st.columns([1, 2])

    with left:
        st.markdown(f"**{len(games_f):,} games** match filters")
        game_table = games_f[["game_id", "winner_team", "last_day", "n_survived"]].copy()
        game_table.columns = ["Game ID", "Winner", "Days", "Survivors"]

        sel = st.dataframe(
            game_table.reset_index(drop=True),
            use_container_width=True,
            height=380,
            on_select="rerun",
            selection_mode="single-row",
        )
        sel_rows = sel.get("selection", {}).get("rows", [])
        gid = int(games_f.iloc[sel_rows[0]]["game_id"]) if sel_rows else int(games_f.iloc[0]["game_id"])

    with right:
        ginfo = games[games["game_id"] == gid].iloc[0]
        emoji = "🏘️" if ginfo["winner_team"] == "Villagers" else "🐺"
        st.markdown(f"### Game `{gid}`  {emoji} **{ginfo['winner_team']} won**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Duration", f"Day {int(ginfo['last_day'])}")
        c2.metric("Players",  int(ginfo["n_players"]))
        c3.metric("Survivors", int(ginfo["n_survived"]))

        # Player roster
        gplayers = players[players["game_id"] == gid].copy()
        roster = gplayers[["player_id", "role", "model_name", "alive_end",
                            "n_messages", "avg_text_len", "n_votes_received",
                            "n_votes_cast"]].copy()
        roster.columns = ["Player", "Role", "Model", "Alive",
                          "Msgs", "Avg Len", "Votes↓", "Votes↑"]
        roster["Alive"] = roster["Alive"].map({True: "✅", False: "❌"})

        def _row_style(row):
            c = ROLE_COLORS.get(row["Role"], "#ffffff") + "30"
            return [f"background-color: {c}"] * len(row)

        st.dataframe(
            roster.set_index("Player").style.apply(_row_style, axis=1),
            use_container_width=True,
        )

    # ── Vote timeline ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🗳️ Vote Timeline")

    gvotes = vote_ev[vote_ev["game_id"] == gid].copy()
    days   = sorted(gvotes["day"].dropna().unique().astype(int))

    def _vote_graph(game_players, votes, title, night=False, day=0):
        """Draw a circular vote graph with arrows voter → target."""
        import math
        player_list = game_players["player_id"].tolist()
        n = len(player_list)
        if n == 0 or len(votes) == 0:
            return None

        angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
        pos = {p: (math.cos(a) * 0.85, math.sin(a) * 0.85)
               for p, a in zip(player_list, angles)}

        role_map = dict(zip(game_players["player_id"], game_players["role"]))

        # Compute alive status at this specific phase, not at game end.
        # Game sequence within each day: day-vote comes before night-vote.
        # seq: Night-0=0, Day1-day=1, Day1-night=2, Day2-day=3, Day2-night=4 ...
        def _vote_seq(d, is_night):
            return 0 if d == 0 else d * 2 + (0 if is_night else -1)

        def _elim_seq(row):
            ed = row.get("eliminated_during_day")
            ep = str(row.get("eliminated_during_phase", "")).lower()
            if pd.isna(ed):
                return float("inf")
            d = int(ed)
            return 0 if d == 0 else d * 2 + (0 if ep == "night" else -1)

        v_seq = _vote_seq(day, night)
        alive_map = {}
        for _, pr in game_players.iterrows():
            pid = pr["player_id"]
            if pr.get("alive_end", False):
                alive_map[pid] = True
            else:
                alive_map[pid] = _elim_seq(pr) >= v_seq

        NODE_COLOR = {
            "Villager": "#e8cc7a", "Werewolf": "#e05252",
            "Seer": "#a9a7c7",     "Doctor": "#e48375",
        }

        fig = go.Figure()

        # Count votes per (voter, target) pair to handle duplicates
        vote_counts = votes.groupby(["actor_id", "target_id"]).size().reset_index(name="n")

        # Arrows
        for _, row in vote_counts.iterrows():
            voter, target = row["actor_id"], row["target_id"]
            if voter not in pos or target not in pos:
                continue
            x0, y0 = pos[voter]
            x1, y1 = pos[target]
            # slightly shorten arrow endpoint so it stops at node edge
            dx, dy = x1 - x0, y1 - y0
            dist = max((dx**2 + dy**2) ** 0.5, 1e-6)
            x1s = x1 - dx / dist * 0.13
            y1s = y1 - dy / dist * 0.13
            fig.add_annotation(
                x=x1s, y=y1s, ax=x0, ay=y0,
                axref="x", ayref="y",
                arrowhead=3, arrowsize=1.5,
                arrowwidth=2.5 if row["n"] > 1 else 1.8,
                arrowcolor="#cc3333" if night else "#4a4a4a",
                showarrow=True, text="",
            )

        # Nodes — label shows "Name (Role)" so identity is unambiguous
        for player in player_list:
            x, y = pos[player]
            role  = role_map.get(player, "Villager")
            alive = alive_map.get(player, True)
            is_wolf = role == "Werewolf"
            color = NODE_COLOR.get(role, "#cccccc")

            # alive → filled circle with role color
            # eliminated → grey, smaller, × symbol
            node_color  = color if alive else "#cccccc"
            border_col  = "#cc0000" if is_wolf else "#777777"
            border_w    = 3.5 if is_wolf else 1.5
            node_size   = 36 if alive else 22
            node_symbol = "circle" if alive else "x"

            role_short = {"Villager": "Vil", "Werewolf": "Wolf",
                          "Seer": "Seer", "Doctor": "Doc"}.get(role, role)
            prefix = "🐺 " if is_wolf else ""
            # two-line label: name on top, role tag below
            label = f"{prefix}{player}<br><sup>{role_short}</sup>"
            label_color = "#cc0000" if is_wolf else ("#555" if not alive else "#222")

            hover = (f"<b>{player}</b><br>"
                     f"Role: <b>{role}</b><br>"
                     f"Status: {'✅ Alive' if alive else '❌ Eliminated'}")

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=node_size, color=node_color,
                            line=dict(width=border_w, color=border_col),
                            symbol=node_symbol),
                text=[label],
                textposition="top center",
                textfont=dict(size=11, color=label_color),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            ))

        # Legend: role colours (circles) + alive/eliminated symbols
        for role, color in NODE_COLOR.items():
            prefix = "🐺 " if role == "Werewolf" else ""
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=13, color=color,
                            line=dict(width=2 if role == "Werewolf" else 1,
                                      color="#cc0000" if role == "Werewolf" else "#888"),
                            symbol="circle"),
                name=f"{prefix}{role}",
                showlegend=True,
            ))
        # alive vs eliminated shape legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=13, color="#aaaaaa", symbol="circle"),
            name="● Alive", showlegend=True))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=13, color="#cccccc", symbol="x"),
            name="✕ Eliminated", showlegend=True))

        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False,
                       showticklabels=False),
            yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False,
                       showticklabels=False),
            height=420, template="plotly_white",
            margin=dict(t=45, b=5, l=5, r=5),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5,
                        xanchor="center", font=dict(size=11),
                        itemsizing="constant"),
        )
        return fig

    if days:
        day_tabs = st.tabs([f"Night 0" if d == 0 else f"Day {d}" for d in days])
        for dtab, day in zip(day_tabs, days):
            with dtab:
                dv = gvotes[gvotes["day"] == day]
                col_d, col_n = st.columns(2)

                for col, vtype, label, is_night in [
                    (col_d, "day",   "☀️ Day Vote — Exile",      False),
                    (col_n, "night", "🌙 Night Vote — Wolf kill", True),
                ]:
                    with col:
                        pv = dv[dv["vote_type"] == vtype]
                        st.markdown(f"**{label}**")
                        if len(pv):
                            fig_v = _vote_graph(gplayers, pv, label, night=is_night, day=day)
                            if fig_v:
                                st.plotly_chart(fig_v, use_container_width=True, key=f"v_{gid}_{day}_{vtype}")
                            # Reasoning expander
                            with st.expander("📝 Reasoning detail"):
                                for _, r in pv.iterrows():
                                    st.markdown(
                                        f"**{r['actor_id']} → {r['target_id']}**  \n"
                                        f"{r['reasoning']}"
                                    )
                                    st.divider()
                        else:
                            st.caption(f"No {vtype} votes this round")
    else:
        st.info("No vote data for this game.")

    # ── Message log ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💬 Public Messages")
    if messages is not None:
        gmsgs = (messages[messages["game_id"] == gid]
                 .merge(gplayers[["player_id", "role"]],
                        left_on="speaker_id", right_on="player_id", how="left")
                 .drop(columns="player_id")
                 .sort_values(["day", "created_at"]))

        if len(gmsgs):
            avail_days = sorted(gmsgs["day"].dropna().unique().astype(int))
            msg_tabs   = st.tabs([f"Day {d}" for d in avail_days])

            for mtab, day in zip(msg_tabs, avail_days):
                with mtab:
                    day_msgs = gmsgs[gmsgs["day"] == day].copy()
                    day_msgs["_tlen"] = day_msgs["text"].str.len().fillna(0).astype(int)
                    for _, row in day_msgs.iterrows():
                        role    = str(row.get("role", ""))
                        color   = ROLE_COLORS.get(role, "#dddddd")
                        speaker = row["speaker_id"]
                        phase   = str(row.get("phase", ""))
                        text    = str(row["text"])
                        tlen    = int(row["_tlen"])
                        safe_text = _html.escape(text).replace("\n", "<br>")
                        st.markdown(
                            f'<div style="background:{color}22; border-left:4px solid {color};'
                            f' padding:8px 14px; border-radius:4px; margin-bottom:8px;">'
                            f'<div style="margin-bottom:5px; display:flex; align-items:center; gap:8px;">'
                            f'<b>{speaker}</b>'
                            f'<span style="background:{color}; color:#fff; font-size:0.76em;'
                            f' padding:2px 8px; border-radius:10px; font-weight:600;">{role}</span>'
                            f'<span style="color:#888; font-size:0.8em">{phase}</span>'
                            f'<span style="margin-left:auto; color:#999; font-size:0.78em">{tlen} chars</span>'
                            f'</div>'
                            f'<div style="font-size:0.92em; line-height:1.55; color:#222;">{safe_text}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.info("No messages found for this game.")
    else:
        st.info(
            "💡 **Message text not loaded.** Export `public_messages.parquet` from the server "
            "as `analysis/speech_analysis/Outputs/tables/public_messages.csv` and rerun the app."
        )

    # ── Elimination order ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚰️ Elimination Order")
    elim = (gplayers[gplayers["alive_end"] == False]
            .sort_values("eliminated_during_day")
            [["player_id", "role", "model_name",
              "eliminated_during_day", "eliminated_during_phase"]]
            .copy())
    elim.columns = ["Player", "Role", "Model", "Day", "Phase"]
    if len(elim):
        st.dataframe(elim, use_container_width=True, hide_index=True)
    else:
        st.success("Nobody was eliminated — unexpected result.")


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — STATISTICS
# ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Player Statistics")
    n_games = df["game_id"].nunique()
    st.caption(f"Showing **{len(df):,} players** across **{n_games:,} games** (filters applied)")

    chart = st.selectbox("Choose chart", [
        "Survival Rate by Role",
        "Survival Rate by LLM Model",
        "Message Count Distribution",
        "Avg Message Length by Role",
        "Votes Received by Role",
        "Votes Cast by Role",
        "First-Day Messages vs Survival",
        "Role × Model Survival Heatmap",
        "Seer Inspection Success Rate",
        "Doctor Heal Success Rate",
        "Werewolf Night-Vote Count",
    ])

    if chart == "Survival Rate by Role":
        agg = df.groupby("role")["alive_end"].mean().reset_index()
        agg.columns = ["Role", "Survival Rate"]
        fig = px.bar(agg, x="Role", y="Survival Rate", color="Role",
                     color_discrete_map=ROLE_COLORS,
                     text=agg["Survival Rate"].map("{:.1%}".format),
                     title="Survival Rate by Role")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], showlegend=False)

    elif chart == "Survival Rate by LLM Model":
        agg = (df.groupby("model_name")["alive_end"].mean()
               .reset_index().sort_values("alive_end", ascending=False))
        fig = px.bar(agg, x="model_name", y="alive_end",
                     text=agg["alive_end"].map("{:.1%}".format),
                     title="Survival Rate by LLM Model",
                     labels={"model_name": "Model", "alive_end": "Survival Rate"},
                     color_discrete_sequence=["#7fb8b0"])
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])

    elif chart == "Message Count Distribution":
        fig = px.histogram(df, x="n_messages", color="role",
                           color_discrete_map=ROLE_COLORS,
                           barmode="overlay", opacity=0.72, nbins=20,
                           title="Message Count Distribution by Role",
                           labels={"n_messages": "Messages per Game"})

    elif chart == "Avg Message Length by Role":
        fig = px.box(df, x="role", y="avg_text_len", color="role",
                     color_discrete_map=ROLE_COLORS, points="outliers",
                     title="Avg Message Length by Role",
                     labels={"avg_text_len": "Avg Length (chars)", "role": "Role"})
        fig.update_layout(showlegend=False)

    elif chart == "Votes Received by Role":
        fig = px.box(df, x="role", y="n_votes_received", color="role",
                     color_discrete_map=ROLE_COLORS, points="outliers",
                     title="Votes Received by Role",
                     labels={"n_votes_received": "Votes Received"})
        fig.update_layout(showlegend=False)

    elif chart == "Votes Cast by Role":
        fig = px.box(df, x="role", y="n_votes_cast", color="role",
                     color_discrete_map=ROLE_COLORS, points="outliers",
                     title="Votes Cast by Role",
                     labels={"n_votes_cast": "Votes Cast"})
        fig.update_layout(showlegend=False)

    elif chart == "First-Day Messages vs Survival":
        fig = px.box(df, x="alive_end", y="first_day_messages",
                     color="alive_end",
                     color_discrete_map={True: "#e48375", False: "#7fb8b0"},
                     title="Day-1 Messages vs Survival",
                     labels={"first_day_messages": "Day 1 Messages", "alive_end": "Survived"},
                     points="outliers")
        fig.update_xaxes(tickvals=[True, False], ticktext=["Survived", "Eliminated"])
        fig.update_layout(showlegend=False)

    elif chart == "Role × Model Survival Heatmap":
        pivot = df.groupby(["role", "model_name"])["alive_end"].mean().unstack()
        fig = px.imshow(pivot, text_auto=".2f",
                        color_continuous_scale="RdYlGn", zmin=0, zmax=1,
                        title="Survival Rate: Role × LLM Model",
                        labels={"color": "Survival Rate"})

    elif chart == "Seer Inspection Success Rate":
        seers = df[df["role"] == "Seer"].copy()
        fig = px.histogram(seers, x="inspect_success_rate", nbins=15,
                           color="alive_end",
                           color_discrete_map={True: "#e48375", False: "#7fb8b0"},
                           barmode="overlay", opacity=0.75,
                           title="Seer Inspection Success Rate (wolf-found / total inspections)",
                           labels={"inspect_success_rate": "Success Rate",
                                   "alive_end": "Survived"})

    elif chart == "Doctor Heal Success Rate":
        docs = df[df["role"] == "Doctor"].copy()
        fig = px.histogram(docs, x="heal_success_rate", nbins=15,
                           color="alive_end",
                           color_discrete_map={True: "#e48375", False: "#7fb8b0"},
                           barmode="overlay", opacity=0.75,
                           title="Doctor Heal Success Rate",
                           labels={"heal_success_rate": "Success Rate",
                                   "alive_end": "Survived"})

    elif chart == "Werewolf Night-Vote Count":
        wolves = df[df["role"] == "Werewolf"].copy()
        fig = px.histogram(wolves, x="n_wolf_votes", nbins=12,
                           color="alive_end",
                           color_discrete_map={True: "#7fb8b0", False: "#e48375"},
                           barmode="overlay", opacity=0.75,
                           title="Werewolf Night-Vote Participation",
                           labels={"n_wolf_votes": "Night Votes Cast",
                                   "alive_end": "Survived"})
    else:
        fig = go.Figure()

    fig.update_layout(height=460, template="plotly_white",
                      font=dict(family="Arial", size=13))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Summary Statistics Table"):
        num_cols = [c for c in METRIC_META if c in df.columns]
        st.dataframe(df[num_cols].describe().round(3), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
# TAB 3 — COMPARISON
# ──────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🔬 Group Comparison")
    st.caption("Compare two player groups on any behavioral metric.")

    cc1, cc2 = st.columns(2)
    with cc1:
        group_by = st.selectbox("Split by", ["Survival", "Winner Team", "Role", "LLM Model"])
    with cc2:
        metric = st.selectbox("Metric", list(METRIC_META.keys()),
                              format_func=lambda k: METRIC_META[k])

    if group_by == "Survival":
        df["_group"] = df["alive_end"].map({True: "Survived", False: "Eliminated"})
        cmap = {"Survived": "#e48375", "Eliminated": "#7fb8b0"}
    elif group_by == "Winner Team":
        df["_group"] = df["winner_team"]
        cmap = WIN_COLORS
    elif group_by == "Role":
        df["_group"] = df["role"]
        cmap = ROLE_COLORS
    else:
        df["_group"] = df["model_name"]
        cmap = {}

    col_box, col_hist = st.columns(2)

    with col_box:
        fig_b = px.box(df, x="_group", y=metric, color="_group",
                       color_discrete_map=cmap, points="outliers",
                       title=f"{METRIC_META[metric]} by {group_by}",
                       labels={metric: METRIC_META[metric], "_group": group_by})
        fig_b.update_layout(showlegend=False, template="plotly_white", height=420)
        st.plotly_chart(fig_b, use_container_width=True)

    with col_hist:
        fig_h = px.histogram(df, x=metric, color="_group",
                             color_discrete_map=cmap,
                             barmode="overlay", opacity=0.72, nbins=25,
                             title=f"Distribution of {METRIC_META[metric]}",
                             labels={metric: METRIC_META[metric], "_group": group_by})
        fig_h.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("#### Summary Statistics by Group")
    summary = (df.groupby("_group")[metric]
               .agg(count="count", mean="mean", median="median", std="std",
                    min="min", max="max")
               .round(3))
    st.dataframe(summary, use_container_width=True)

    # scatter: two numeric metrics, coloured by group
    st.markdown("---")
    st.markdown("#### Scatter: any two metrics")
    sx, sy = st.columns(2)
    with sx:
        x_col = st.selectbox("X axis", list(METRIC_META.keys()), index=0,
                             format_func=lambda k: METRIC_META[k], key="sx")
    with sy:
        y_col = st.selectbox("Y axis", list(METRIC_META.keys()), index=5,
                             format_func=lambda k: METRIC_META[k], key="sy")

    fig_sc = px.scatter(df, x=x_col, y=y_col, color="_group",
                        color_discrete_map=cmap, opacity=0.55,
                        hover_data=["player_id", "role", "model_name"],
                        title=f"{METRIC_META[x_col]} vs {METRIC_META[y_col]}",
                        labels={x_col: METRIC_META[x_col],
                                y_col: METRIC_META[y_col],
                                "_group": group_by},
                        trendline="ols")
    fig_sc.update_layout(template="plotly_white", height=440)
    st.plotly_chart(fig_sc, use_container_width=True)
