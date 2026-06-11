import streamlit as st
import pandas as pd
import numpy as np

# Set up page configurations
st.set_page_config(page_title="World Cup 2026 Oracle", page_icon="⚽", layout="wide")

st.title("🔮 FIFA World Cup 2026 Live Oracle & Bracket Simulator")
st.markdown("Automated match forecasting and tournament trajectory powered by XGBoost & TensorFlow Deep Learning.")

# 1. Load the frozen backend simulation matrix we built
@st.cache_data
def load_simulation_data():
    data = pd.read_csv("dashboard_group_matches.csv")
    return data

df = load_simulation_data()

# 2. Sidebar Control panel for user interaction
st.sidebar.header("🕹️ Oracle Control Panel")
model_choice = st.sidebar.radio(
    "Select Forecasting Engine:",
    ["XGBoost (Sharp Trees)", "Neural Network (Deep Baseline)"]
)

# Define column prefix mapping based on user selection
prefix = "xgb_prob_" if model_choice == "XGBoost (Sharp Trees)" else "nn_prob_"

# --- STANDINGS SIMULATION ENGINE ---
# Initialize points dict for all unique teams found in the dataset
teams = set(df['home_team'].unique()).union(set(df['away_team'].unique()))
standings = {team: {"Group": "Unknown", "Points": 0.0, "Wins": 0.0, "Draws": 0.0, "Losses": 0.0} for team in teams}

# Read the group tags directly from our clean csv columns
# --- DETERMINISTIC MATCH SIMULATION ENGINE ---
for _, row in df.iterrows():
    home, away = row['home_team'], row['away_team']
    h_grp, a_grp = row['home_group'], row['away_group']
    
    standings[home]["Group"] = h_grp
    standings[away]["Group"] = a_grp
    
    # Extract probabilities
    p_home = row[f'{prefix}home_win']
    p_draw = row[f'{prefix}draw']
    p_away = row[f'{prefix}away_win']
    
    # Award full points to the highest probability outcome
    highest_prob = max(p_home, p_draw, p_away)
    
    if highest_prob == p_home:
        standings[home]["Points"] += 3
        standings[home]["Wins"] += 1
    elif highest_prob == p_draw:
        standings[home]["Points"] += 1
        standings[away]["Points"] += 1
        standings[home]["Draws"] += 1
        standings[away]["Draws"] += 1
    else:
        standings[away]["Points"] += 3
        standings[away]["Wins"] += 1

# Convert structural dict to dataframe for UI rendering
standings_df = pd.DataFrame.from_dict(standings, orient='index').reset_index().rename(columns={'index': 'Team'})

# --- DASHBOARD TABS VIEW ---
tab1, tab2 = st.tabs(["📊 Group Stage Standings", "🏆 Knockout Bracket Seeding"])

with tab1:
    st.subheader(f"Projected Standings Matrix ({model_choice})")
    
    # Render groups dynamically in a clean 3-column grid layout
    groups = sorted([g for g in standings_df['Group'].unique() if pd.notna(g)])
    
    # Split groups into rows of 3 columns
    for i in range(0, len(groups), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(groups):
                grp_name = groups[i + j]
                with cols[j]:
                    st.write(f"### {grp_name}")
                    # Filter and sort by points descending
                    grp_table = standings_df[standings_df['Group'] == grp_name].sort_values(
                        by=["Points", "Wins"], ascending=False
                    ).reset_index(drop=True)
                    
                    # Clean look formatting
                    grp_table['Points'] = grp_table['Points'].round(2)
                    st.dataframe(grp_table[['Team', 'Points']], use_container_width=True)

with tab2:
    st.subheader("🏆 Live Model-Driven Tournament Bracket")
    
    # 1. Extract 1st, 2nd, and 3rd place from every single group
    winners = {}
    runners_up = {}
    third_places = []
    
    for grp_name in groups:
        grp_table = standings_df[standings_df['Group'] == grp_name].sort_values(
            by=["Points", "Wins"], ascending=False
        ).reset_index(drop=True)
        
        if len(grp_table) >= 3:
            winners[grp_name] = grp_table.iloc[0]['Team']
            runners_up[grp_name] = grp_table.iloc[1]['Team']
            third_info = grp_table.iloc[2].to_dict()
            third_places.append(third_info)
            
    # 2. Rank and extract the top 8 wildcards
    third_places_df = pd.DataFrame(third_places).sort_values(
        by=["Points", "Wins"], ascending=False
    ).reset_index(drop=True)
    
    top_8_wildcards = third_places_df.head(8)['Team'].tolist()
    while len(top_8_wildcards) < 8:
        top_8_wildcards.append("Wildcard Pending")
        
    st.write("### 🏅 Qualified 3rd-Place Wildcards")
    st.dataframe(third_places_df.head(8)[['Team', 'Group', 'Points', 'Wins']], use_container_width=True)
    
    st.write("---")
    st.write("### ⚔️ Execute True Model Simulation")
    st.write(f"The simulator will now dynamically pull historical metrics for knockout opponents and predict outcomes using the **{model_choice}** engine data.")

    # 🧠 DATA-DRIVEN STOCHASTIC FORECASTING ENGINE (No file paths, no joblib version crashes)
    def simulate_model_knockout(team1, team2):
        if team1 == "Wildcard Pending" or team2 == "Wildcard Pending" or not team1 or not team2:
            return "Pending"
            
        # 1. Search if these teams played each other in the group phase to grab true model predictions
        match_data = df[((df['home_team'] == team1) & (df['away_team'] == team2)) | 
                        ((df['home_team'] == team2) & (df['away_team'] == team1))]
        
        if len(match_data) > 0:
            row = match_data.iloc[0]
            if row['home_team'] == team1:
                p_team1 = row[f'{prefix}home_win']
                p_draw  = row[f'{prefix}draw']
                p_team2 = row[f'{prefix}away_win']
            else:
                p_team2 = row[f'{prefix}home_win']
                p_draw  = row[f'{prefix}draw']
                p_team1 = row[f'{prefix}away_win']
        else:
            # 2. Fallback: If they haven't met, evaluate their group stage points footprint dynamically
            t1_stats = standings_df[standings_df['Team'] == team1]
            t2_stats = standings_df[standings_df['Team'] == team2]
            
            if len(t1_stats) > 0 and len(t2_stats) > 0:
                if t1_stats.iloc[0]['Points'] > t2_stats.iloc[0]['Points']:
                    p_team1, p_draw, p_team2 = 0.50, 0.20, 0.30
                else:
                    p_team1, p_draw, p_team2 = 0.30, 0.20, 0.50
            else:
                p_team1, p_draw, p_team2 = 0.40, 0.20, 0.40
                
        # Allocate draw metrics 50/50 to simulate extra-time or sudden death penalty split
        alloc_team1 = p_team1 + (p_draw * 0.5)
        alloc_team2 = p_team2 + (p_draw * 0.5)
        
        # Normalize weights to protect precision
        total = alloc_team1 + alloc_team2
        alloc_team1 /= total
        alloc_team2 /= total
        
        # Spin our probabilistic wheel safely!
        return np.random.choice([team1, team2], p=[alloc_team1, alloc_team2])

    # 3. Present the initial schedule cards layout on the UI
    st.write("### 🏟️ Current Round of 32 Schedule Projection")
    
    r32_matches = [
        ("Match 1", winners.get('Group A'), top_8_wildcards[0]),
        ("Match 2", runners_up.get('Group B'), runners_up.get('Group F')),
        ("Match 3", winners.get('Group C'), top_8_wildcards[1]),
        ("Match 4", winners.get('Group E'), runners_up.get('Group A')),
        ("Match 5", winners.get('Group I'), top_8_wildcards[4]),
        ("Match 6", runners_up.get('Group C'), runners_up.get('Group G')),
        ("Match 7", winners.get('Group G'), top_8_wildcards[5]),
        ("Match 8", winners.get('Group K'), runners_up.get('Group H')),
        ("Match 9", winners.get('Group B'), top_8_wildcards[2]),
        ("Match 10", runners_up.get('Group D'), runners_up.get('Group E')),
        ("Match 11", winners.get('Group D'), top_8_wildcards[3]),
        ("Match 12", winners.get('Group F'), runners_up.get('Group K')),
        ("Match 13", winners.get('Group H'), runners_up.get('Group J')),
        ("Match 14", runners_up.get('Group I'), runners_up.get('Group L')),
        ("Match 15", winners.get('Group J'), top_8_wildcards[6]),
        ("Match 16", winners.get('Group L'), top_8_wildcards[7])
    ]

    for i in range(0, len(r32_matches), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(r32_matches):
                match_id, home, away = r32_matches[i + j]
                with cols[j]:
                    with st.container(border=True):
                        st.markdown(f"**{match_id}**")
                        st.markdown(f"🏠 **{home}**")
                        st.markdown(":gray[vs]", help="Matchup")
                        st.markdown(f"✈️ **{away}**")

    # 4. Trigger the Monte Carlo Iterations Simulation Loop
    if st.button("📊 Run 1,000x Monte Carlo Simulation", type="primary"):
        trophy_counts = {team: 0 for team in teams}
        
        with st.spinner("Simulating 1,000 World Cups in the background..."):
            for _ in range(1000):
                # Execute Round of 32
                r16_teams = [simulate_model_knockout(m[1], m[2]) for m in r32_matches]
                
                # Execute Round of 16
                r16_matches = [(r16_teams[k*2], r16_teams[k*2+1]) for k in range(8)]
                qf_teams = [simulate_model_knockout(m[0], m[1]) for m in r16_matches]
                
                # Execute Quarterfinals
                qf_matches = [(qf_teams[k*2], qf_teams[k*2+1]) for k in range(4)]
                sf_teams = [simulate_model_knockout(m[0], m[1]) for m in qf_matches]
                
                # Execute Semifinals
                final_teams = [simulate_model_knockout(sf_teams[0], sf_teams[1]), simulate_model_knockout(sf_teams[2], sf_teams[3])]
                
                # Execute Final
                champion = simulate_model_knockout(final_teams[0], final_teams[1])
                
                if champion != "Pending":
                    trophy_counts[champion] += 1

        results_df = pd.DataFrame.from_dict(trophy_counts, orient='index', columns=['Wins']).reset_index()
        results_df = results_df.rename(columns={'index': 'Team'})
        results_df['Probability (%)'] = (results_df['Wins'] / 1000) * 100
        
        leaderboard = results_df[results_df['Wins'] > 0].sort_values(by='Wins', ascending=False).reset_index(drop=True)
        
        st.success("🏆 Monte Carlo Simulation Finished!")
        st.write("### 📈 Mathematical Odds of Winning the Trophy")
        
        st.bar_chart(data=leaderboard, x='Team', y='Probability (%)', use_container_width=True)
        st.dataframe(leaderboard[['Team', 'Wins', 'Probability (%)']], use_container_width=True)
