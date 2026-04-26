"""
Wheelchair Rugby Analysis: Canada Lineup Optimization
Streamlit Web Application

This application analyzes wheelchair rugby line-up data to optimize Team Canada's 
player combinations. It finds the best 4-player lineups (constrained by the 8.0 
classification point limit) that maximize Goal Differential relative to opponents.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from itertools import combinations

# Set page configuration
st.set_page_config(
    page_title="Wheelchair Rugby Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2C5282;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load stint and player data from CSV files."""
    try:
        stints = pd.read_csv('stint_data.csv')
        players = pd.read_csv('player_data.csv')
        return stints, players
    except FileNotFoundError:
        st.error("Error: CSV files not found. Please ensure 'stint_data.csv' and 'player_data.csv' are in the same directory.")
        return None, None


@st.cache_data
def preprocess_data(stints):
    """Filter artifacts and calculate metrics."""
    stints = stints[stints['minutes'] > 0.01].copy()
    stints['goal_diff'] = stints['h_goals'] - stints['a_goals']
    stints['goal_diff_per_min'] = stints['goal_diff'] / stints['minutes']
    stints['h_goals_per_min'] = stints['h_goals'] / stints['minutes']
    stints['a_goals_per_min'] = stints['a_goals'] / stints['minutes']
    return stints


@st.cache_data
def train_global_model(stints):
    """Train global player rating model using Ridge Regression."""
    home_cols = ['home1', 'home2', 'home3', 'home4']
    away_cols = ['away1', 'away2', 'away3', 'away4']
    
    # Get all unique players
    all_players = set()
    for col in home_cols + away_cols:
        all_players.update(stints[col].unique())
    all_players = sorted(list(all_players))
    player_to_idx = {p: i for i, p in enumerate(all_players)}
    
    # Construct feature matrix
    data = []
    for idx, row in stints.iterrows():
        row_data = np.zeros(len(all_players))
        for col in home_cols:
            if row[col] in player_to_idx:
                row_data[player_to_idx[row[col]]] = 1.0
        for col in away_cols:
            if row[col] in player_to_idx:
                row_data[player_to_idx[row[col]]] = -1.0
        data.append(row_data)
    
    X = pd.DataFrame(data, columns=all_players, index=stints.index)
    y = stints['goal_diff_per_min']
    weights = stints['minutes']
    
    # Train model
    ridge = Ridge(alpha=1.0, fit_intercept=False)
    ridge.fit(X, y, sample_weight=weights)
    
    player_net_ratings = pd.Series(ridge.coef_, index=all_players)
    return player_net_ratings, all_players


@st.cache_data
def train_canada_model(stints, ratings_dict):
    """Train Canada-specific scoring model."""
    home_cols = ['home1', 'home2', 'home3', 'home4']
    away_cols = ['away1', 'away2', 'away3', 'away4']
    
    def get_opp_rating_sum(row, is_canada_home):
        opp_cols = away_cols if is_canada_home else home_cols
        return sum(ratings_dict.get(row[c], 0) for c in opp_cols)
    
    def get_canada_players(row, is_canada_home):
        cols = home_cols if is_canada_home else away_cols
        return [row[c] for c in cols]
    
    # Filter Canada games
    canada_home = stints[stints['h_team'] == 'Canada'].copy()
    canada_away = stints[stints['a_team'] == 'Canada'].copy()
    
    canada_home['is_home'] = 1
    canada_home['canada_goals_pm'] = canada_home['h_goals_per_min']
    canada_home['opp_strength'] = canada_home.apply(lambda r: get_opp_rating_sum(r, True), axis=1)
    
    canada_away['is_home'] = 0
    canada_away['canada_goals_pm'] = canada_away['a_goals_per_min']
    canada_away['opp_strength'] = canada_away.apply(lambda r: get_opp_rating_sum(r, False), axis=1)
    
    combined_can = pd.concat([canada_home, canada_away])
    
    # Identify Canada unique players
    can_players_set = set()
    for cols in [home_cols, away_cols]:
        for p in combined_can[cols].values.flatten():
            if isinstance(p, str) and "Canada" in p:
                can_players_set.add(p)
    can_players_list = sorted(list(can_players_set))
    
    # Build feature matrix
    X_can_data = []
    for idx, row in combined_can.iterrows():
        feats = {p: 0 for p in can_players_list}
        current_sq = get_canada_players(row, row['is_home'] == 1)
        for p in current_sq:
            if p in feats:
                feats[p] = 1
        feats['opp_strength'] = row['opp_strength']
        X_can_data.append(feats)
    
    X_can = pd.DataFrame(X_can_data)
    y_can = combined_can['canada_goals_pm']
    w_can = combined_can['minutes']
    
    # Train model
    can_model = Ridge(alpha=1.0, fit_intercept=True)
    can_model.fit(X_can, y_can, sample_weight=w_can)
    
    return can_model, can_players_list, X_can.columns.tolist()


def main():
    # Header
    st.markdown('<p class="main-header">Wheelchair Rugby Analysis: Canada Lineup Optimization</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This application analyzes wheelchair rugby line-up data to optimize Team Canada's player combinations. 
    We aim to find the best 4-player lineups (constrained by the 8.0 classification point limit) that 
    maximize Goal Differential relative to opponents.
    """)
    
    # Load data
    stints, players = load_data()
    if stints is None or players is None:
        return
    
    stints = preprocess_data(stints)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Overview & EDA", "Player Ratings", "Lineup Optimizer", "Player Rankings"]
    )
    
    # Display metrics in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Summary")
    st.sidebar.metric("Total Stints", f"{len(stints):,}")
    st.sidebar.metric("Total Players", f"{len(players):,}")
    st.sidebar.metric("Unique Teams", f"{stints['h_team'].nunique()}")
    
    if page == "Overview & EDA":
        display_eda_page(stints)
    elif page == "Player Ratings":
        display_player_ratings_page(stints)
    elif page == "Lineup Optimizer":
        display_lineup_optimizer_page(stints, players)
    elif page == "Player Rankings":
        display_player_rankings_page(stints, players)


def display_eda_page(stints):
    """Display Exploratory Data Analysis page."""
    st.markdown('<p class="sub-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Stint Duration Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(stints['minutes'], bins=50, kde=True, color='skyblue', ax=ax)
        ax.set_title('Distribution of Stint Durations (Minutes)')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        plt.close()
        
        st.info(f"""
        **Statistics:**
        - Mean Stint Duration: {stints['minutes'].mean():.2f} min
        - Median Stint Duration: {stints['minutes'].median():.2f} min
        """)
    
    with col2:
        st.markdown("### Team Performance Summary")
        # Calculate Home Performance
        home_perf = stints.groupby('h_team')['goal_diff_per_min'].mean()
        away_perf = stints.groupby('a_team')['goal_diff_per_min'].mean() * -1
        
        team_stats = pd.DataFrame({'Home': home_perf, 'Away': away_perf})
        team_stats['Total'] = team_stats['Home'].fillna(0) + team_stats['Away'].fillna(0)
        team_stats = team_stats.sort_values(by='Total', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        team_stats[['Home', 'Away']].plot(kind='bar', stacked=False, ax=ax, color=['skyblue', 'salmon'], width=0.8)
        ax.set_title('Average Goal Differential per Minute (Home vs Away)')
        ax.set_xlabel('Team')
        ax.set_ylabel('Avg Goal Diff / Minute')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.legend(title='Venue Performance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Canada Performance Highlight
    if 'Canada' in team_stats.index:
        st.markdown("### Canada Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Home Performance", f"{team_stats.loc['Canada', 'Home']:.3f}")
        with col2:
            st.metric("Away Performance", f"{team_stats.loc['Canada', 'Away']:.3f}")
        with col3:
            st.metric("Total Index", f"{team_stats.loc['Canada', 'Total']:.3f}")


def display_player_ratings_page(stints):
    """Display Global Player Ratings page."""
    st.markdown('<p class="sub-header">Global Player Ratings</p>', unsafe_allow_html=True)
    
    st.markdown("""
    We use **Weighted Ridge Regression** to decouple individual player impact from their teammates.
    - **Target**: Goal Differential per Minute
    - **Features**: Sparse matrix of players (Home=+1, Away=-1)
    - **Weights**: Stint duration (Minutes)
    """)
    
    with st.spinner("Training global rating model..."):
        player_net_ratings, all_players = train_global_model(stints)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 15 Players")
        top_players = player_net_ratings.sort_values(ascending=False).head(15)
        top_df = pd.DataFrame({
            'Player': top_players.index,
            'Net Rating': top_players.values
        }).reset_index(drop=True)
        top_df.index += 1
        st.dataframe(top_df, use_container_width=True)
    
    with col2:
        st.markdown("### Bottom 15 Players")
        bottom_players = player_net_ratings.sort_values(ascending=True).head(15)
        bottom_df = pd.DataFrame({
            'Player': bottom_players.index,
            'Net Rating': bottom_players.values
        }).reset_index(drop=True)
        bottom_df.index += 1
        st.dataframe(bottom_df, use_container_width=True)
    
    # Top players visualization
    st.markdown("### Top 20 Players by Net Rating")
    fig, ax = plt.subplots(figsize=(12, 8))
    top_20 = player_net_ratings.sort_values(ascending=False).head(20)
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_20.values]
    bars = ax.barh(range(len(top_20)), top_20.values, color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20.index)
    ax.set_xlabel('Net Rating (Goal Diff per Min)')
    ax.set_title('Top 20 Players by Global Net Rating')
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_lineup_optimizer_page(stints, players):
    """Display Lineup Optimization page."""
    st.markdown('<p class="sub-header">Lineup Optimization</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Generate all valid 4-player combinations for Team Canada that satisfy the classification rule:
    **Total Class Points ≤ 8.0**
    
    Lineups are prioritized by their **Predicted Goals per Minute**.
    """)
    
    # Train models
    with st.spinner("Training models..."):
        player_net_ratings, all_players = train_global_model(stints)
        ratings_dict = player_net_ratings.to_dict()
        can_model, can_players_list, feature_names = train_canada_model(stints, ratings_dict)
    
    class_dict = dict(zip(players['player'], players['rating']))
    player_coefs = {name: can_model.coef_[i] for i, name in enumerate(feature_names)}
    opp_strength_coef = player_coefs.get('opp_strength', 0.0)
    
    # --- Team Canada Configuration ---
    st.markdown("### Team Canada Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Home/Away Toggle
        canada_venue = st.radio(
            "Canada Playing:",
            options=["Home", "Away"],
            horizontal=True,
            help="Select whether Team Canada is playing at home or away"
        )
        is_canada_home = (canada_venue == "Home")
    
    with col2:
        # Injured Players Selection
        canada_player_names = [p.replace("Canada_", "") for p in can_players_list]
        injured_players_display = st.multiselect(
            "Injured/Unavailable Players:",
            options=canada_player_names,
            default=[],
            help="Select players who are injured or unavailable - they will be excluded from lineup generation"
        )
        # Convert back to full player IDs
        injured_players = [f"Canada_{p}" for p in injured_players_display]
    
    if injured_players:
        st.warning(f"**Excluded from lineups:** {', '.join(injured_players_display)}")
    
    st.markdown("---")
    
    # --- Opponent Configuration ---
    st.markdown("### Opponent Configuration")
    
    # Get unique teams from data
    all_teams = sorted(set(stints['h_team'].unique()) | set(stints['a_team'].unique()))
    all_teams = [t for t in all_teams if t != 'Canada']
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_opponent = st.checkbox("Specify Opponent Team", value=True)
        if use_opponent:
            opp_country = st.selectbox("Select Opponent Country", options=all_teams)
        else:
            opp_country = None
    
    current_opp_strength = 0.0
    
    if use_opponent and opp_country:
        with col2:
            st.markdown(f"**Enter 4 player numbers for {opp_country}:**")
            opp_p1 = st.number_input(f"{opp_country} Player 1", min_value=1, max_value=12, value=1)
            opp_p2 = st.number_input(f"{opp_country} Player 2", min_value=1, max_value=12, value=2)
            opp_p3 = st.number_input(f"{opp_country} Player 3", min_value=1, max_value=12, value=3)
            opp_p4 = st.number_input(f"{opp_country} Player 4", min_value=1, max_value=12, value=4)
        
        opp_ids = [f"{opp_country}_p{num}" for num in [opp_p1, opp_p2, opp_p3, opp_p4]]
        current_opp_strength = sum(ratings_dict.get(pid, 0.0) for pid in opp_ids)
        
        st.info(f"""
        **Match Configuration:**
        - Canada: {"Home" if is_canada_home else "Away"}
        - Opponent: {opp_country} ({"Away" if is_canada_home else "Home"})
        - Opponent Players: {', '.join(opp_ids)}
        - Calculated Opponent Strength: {current_opp_strength:.4f}
        """)
    else:
        st.info(f"Canada playing {'Home' if is_canada_home else 'Away'}. No opponent specified - using average opponent strength = 0.")
    
    # Generate and rank lineups
    if st.button("Generate Optimal Lineups", type="primary"):
        with st.spinner("Evaluating all possible lineups..."):
            # Filter out injured players from available squad
            can_squad = [f for f in feature_names if f != 'opp_strength' and f not in injured_players]
            
            if len(can_squad) < 4:
                st.error("Not enough available players to form a lineup. Please reduce the number of injured players.")
            else:
                combos = list(combinations(can_squad, 4))
                valid_lineups = []
                
                for lineup in combos:
                    total_class = sum(class_dict.get(p, 0) for p in lineup)
                    if total_class <= 8.0:
                        can_player_sum = sum(player_coefs.get(p, 0) for p in lineup)
                        
                        # Apply home/away adjustment to prediction
                        # When away, the opponent strength effect may differ
                        if is_canada_home:
                            pred_gpm = can_model.intercept_ + can_player_sum + (current_opp_strength * opp_strength_coef)
                        else:
                            # Away games: opponent strength has different impact
                            pred_gpm = can_model.intercept_ + can_player_sum - (current_opp_strength * opp_strength_coef * 0.5)
                        
                        valid_lineups.append({
                            'lineup': lineup,
                            'total_class': total_class,
                            'pred_goals_per_min': pred_gpm,
                            'pred_goals_per_8min': pred_gpm * 8,
                            'venue': canada_venue
                        })
                
                valid_lineups.sort(key=lambda x: x['pred_goals_per_min'], reverse=True)
                results_df = pd.DataFrame(valid_lineups)
        
                if not results_df.empty:
                    results_df['lineup_clean'] = results_df['lineup'].apply(
                        lambda x: ", ".join([p.replace("Canada_", "") for p in x])
                    )
                    
                    st.success(f"Found {len(results_df)} valid lineups! (Excluding {len(injured_players)} injured player(s))")
                    
                    # Display top 10
                    st.markdown(f"### Top 10 Recommended Lineups ({canada_venue})")
                    display_df = results_df[['lineup_clean', 'total_class', 'pred_goals_per_min', 'pred_goals_per_8min']].head(10)
                    display_df = display_df.rename(columns={
                        'lineup_clean': 'Lineup',
                        'total_class': 'Total Class',
                        'pred_goals_per_min': 'Pred Goals/Min',
                        'pred_goals_per_8min': 'Pred Goals/8 Min'
                    })
                    display_df.index = range(1, len(display_df) + 1)
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Visualization
                    st.markdown("### Top 10 Lineups Visualization")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    top_10 = results_df.head(10)
                    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10)))
                    bars = ax.barh(range(len(top_10)), top_10['pred_goals_per_min'].values, color=colors)
                    ax.set_yticks(range(len(top_10)))
                    ax.set_yticklabels(top_10['lineup_clean'].values)
                    ax.set_xlabel('Predicted Goals per Minute')
                    ax.set_title(f'Top 10 Canada Lineups by Predicted Scoring Rate ({canada_venue})')
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Download option
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download All Lineups (CSV)",
                        data=csv,
                        file_name=f"canada_lineups_{canada_venue.lower()}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No valid lineups found. Check classification data or reduce injured players.")


def display_player_rankings_page(stints, players):
    """Display Player Rankings by Role page."""
    st.markdown('<p class="sub-header">Canada Player Rankings by Role</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Canadian players are ranked based on their **Net Rating (Ridge Regression Coefficient)** 
    and **Raw Plus-Minus per Minute**.
    
    Players are categorized by their classification score:
    - **Attacker**: Class ≥ 2.5
    - **Flex**: 1.0 < Class < 2.5
    - **Defender**: Class ≤ 1.0
    """)
    
    home_cols = ['home1', 'home2', 'home3', 'home4']
    away_cols = ['away1', 'away2', 'away3', 'away4']
    
    # Train model
    with st.spinner("Computing player ratings..."):
        player_net_ratings, all_players = train_global_model(stints)
        ratings_dict = player_net_ratings.to_dict()
    
    class_dict = dict(zip(players['player'], players['rating']))
    
    # Get Canada players
    can_players_list = [p for p in all_players if 'Canada' in p]
    
    # Calculate raw stats
    player_raw_stats = {}
    for p in can_players_list:
        mask = (stints[home_cols + away_cols] == p).any(axis=1)
        p_stints = stints[mask]
        
        total_goal_diff = 0
        total_mins = p_stints['minutes'].sum()
        
        for _, row in p_stints.iterrows():
            gd = row['h_goals'] - row['a_goals']
            if p in row[home_cols].values:
                total_goal_diff += gd
            else:
                total_goal_diff -= gd
        
        raw_pm_per_min = total_goal_diff / total_mins if total_mins > 0 else 0
        player_raw_stats[p] = raw_pm_per_min
    
    # Create ranking dataframe
    rank_data = []
    for p in can_players_list:
        cls = class_dict.get(p, 0.0)
        net_rating = ratings_dict.get(p, 0.0)
        raw_pm = player_raw_stats.get(p, 0.0)
        
        if cls >= 2.5:
            role = 'Attacker'
        elif cls > 1.0:
            role = 'Flex'
        else:
            role = 'Defender'
        
        rank_data.append({
            'Player': p.replace('Canada_', ''),
            'Class': cls,
            'Role': role,
            'Net Rating (Reg)': net_rating,
            'Raw +/- per Min': raw_pm
        })
    
    rank_df = pd.DataFrame(rank_data)
    
    # Display by role
    roles = ['Attacker', 'Flex', 'Defender']
    role_colors = {'Attacker': '#e74c3c', 'Flex': '#f39c12', 'Defender': '#3498db'}
    
    tabs = st.tabs(roles)
    
    for i, role in enumerate(roles):
        with tabs[i]:
            subset = rank_df[rank_df['Role'] == role].sort_values(
                by='Net Rating (Reg)', ascending=False
            ).reset_index(drop=True)
            subset.index += 1
            
            st.dataframe(subset, use_container_width=True)
            
            # Visualization
            if len(subset) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = [role_colors[role]] * len(subset)
                bars = ax.bar(subset['Player'], subset['Net Rating (Reg)'], color=colors, alpha=0.8)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.set_xlabel('Player')
                ax.set_ylabel('Net Rating')
                ax.set_title(f'{role} Rankings by Net Rating')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # Overall comparison
    st.markdown("### Overall Player Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for role in roles:
        subset = rank_df[rank_df['Role'] == role]
        ax.scatter(subset['Net Rating (Reg)'], subset['Raw +/- per Min'], 
                   label=role, color=role_colors[role], s=100, alpha=0.7)
        for _, row in subset.iterrows():
            ax.annotate(row['Player'], (row['Net Rating (Reg)'], row['Raw +/- per Min']),
                       fontsize=9, ha='center', va='bottom')
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Net Rating (Regularized)')
    ax.set_ylabel('Raw +/- per Minute')
    ax.set_title('Canada Players: Net Rating vs Raw Performance')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    main()
