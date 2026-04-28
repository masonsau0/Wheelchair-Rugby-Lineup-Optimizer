# Wheelchair Rugby Lineup Optimizer

**[Live demo](https://mason-wheelchair-rugby-lineup.streamlit.app/)**: runs in the browser, no install required.

Data-driven lineup optimiser for **Team Canada Wheelchair Rugby**. Uses
**regularised plus-minus** regression on **7,448 historical stint
records** to isolate individual player contributions from team effects,
then enumerates every feasible 4-player lineup under the IWRF
**total-classification ≤ 8.0 points** constraint and ranks the top-10 by
predicted goal differential per stint.

## What plus-minus regression actually does

A *stint* is a continuous on-court interval where the lineup doesn't
change. The target is the goal differential during the stint. We fit:

```
goal_diff_per_minute  ~  Σ_p β_p · I(player p on court)  +  Σ_q γ_q · I(opponent q on court)
```

with **Ridge / Lasso** regularisation. The fitted `β_p` is player *p*'s
isolated contribution: the part of the team's per-minute scoring that
travels with that player even after removing the effect of the four
team-mates on court at the same time. Regularisation is essential: the
small-sample-size problem is severe for niche role players, and the
collinearity between team-mates who always play together would blow up
an unregularised fit.

## Approach

1. **Stint-level data**: 7,448 records of (lineup, opponent lineup, stint duration, home goals, away goals).
2. **Stage 1: player ratings.** One-hot encode the 4-of-N home and 4-of-M away players per stint; fit Ridge / Lasso of `goal_diff_per_minute` on the encoded features. Read off `β_p` as each player's rating.
3. **Stage 2: lineup enumeration.** Generate every 4-player combination of the Team Canada roster that satisfies the IWRF classification cap, score each lineup as `Σ β_p` for its members, and return the top 10.
4. **Per-role rankings.** Players are also ranked within their classification band (low-point, mid-point, high-point) so coaches can swap roles cleanly when injuries force changes.

## Result

The best feasible lineup predicts **17.25 goals per stint** goal
differential, with the next nine lineups within ~1 goal of the optimum.
The top combinations cluster around two or three high-impact players plus
complementary low-point partners.

## Repository layout

```
.
├── rugby_analysis_final.ipynb   ← end-to-end analysis notebook
├── rugby_streamlit_app.py       ← interactive Streamlit dashboard
├── player_data.csv              ← roster with IWRF classification & per-game stats
├── stint_data.csv               ← 7,448 stint records (lineup + outcome)
├── requirements.txt
└── README.md
```

## Run it

### Notebook walkthrough

```bash
pip install -r requirements.txt
jupyter notebook rugby_analysis_final.ipynb
```

### Interactive dashboard

```bash
streamlit run rugby_streamlit_app.py
```

The dashboard:

- **Exploratory data analysis**: stint-duration distribution, team-level performance summary, Canada's home/away splits.
- **Global player ratings**: fitted Ridge ratings with confidence bands; top-15 / bottom-15 tables and a top-20 net-rating ranking.
- **Lineup optimization**: pick the venue and opponent, enter the opposing four players, and the dashboard returns Team Canada's top-10 best lineups with predicted goal differential, coloured by classification mix.
- **Per-role rankings**: players ranked within their classification band so coaches can substitute cleanly when injuries force changes.
- **Overall player comparison**: scatter of attacking vs. defensive contribution to spot two-way players.

### Programmatic use

```python
import pandas as pd
from sklearn.linear_model import Ridge
from itertools import combinations

stints = pd.read_csv("stint_data.csv")
players = pd.read_csv("player_data.csv")
# ... see rugby_analysis_final.ipynb for full pipeline
```

## Stack

Python · scikit-learn (Ridge, Lasso) · pandas · NumPy · matplotlib ·
seaborn · **Streamlit** (dashboard)
