import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .win-box {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
    }
    .draw-box {
        background-color: #fff3cd;
        border: 2px solid #ffeaa7;
    }
    .lose-box {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    """Load all models and encoders"""
    try:
        model = joblib.load('football_predictor_model.pkl')
        local_encoder = joblib.load('local_team_encoder.pkl')
        visitor_encoder = joblib.load('visitor_team_encoder.pkl')
        season_encoder = joblib.load('season_encoder.pkl')
        division_encoder = joblib.load('division_encoder.pkl')
        round_encoder = joblib.load('round_encoder.pkl')
        team_stats = joblib.load('team_statistics.pkl')
        
        return model, local_encoder, visitor_encoder, season_encoder, division_encoder, round_encoder, team_stats
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please make sure you have run the Jupyter notebook first to train and save the models!")
        return None, None, None, None, None, None, None

def calculate_team_stats(df):
    """Calculate team statistics from historical data"""
    team_stats = {}
    
    # Calculate stats for each team
    all_teams = list(set(df['localTeam'].unique()) | set(df['visitorTeam'].unique()))
    
    for team in all_teams:
        # Home games (when team plays at home)
        home_games = df[df['localTeam'] == team]
        home_wins = len(home_games[home_games['localGoals'] > home_games['visitorGoals']])
        home_draws = len(home_games[home_games['localGoals'] == home_games['visitorGoals']])
        
        # Away games (when team plays away)
        away_games = df[df['visitorTeam'] == team]
        away_wins = len(away_games[away_games['visitorGoals'] > away_games['localGoals']])
        away_draws = len(away_games[away_games['visitorGoals'] == away_games['localGoals']])
        
        total_games = len(home_games) + len(away_games)
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        
        if total_games > 0:
            win_rate = total_wins / total_games
            draw_rate = total_draws / total_games
        else:
            win_rate = 0
            draw_rate = 0
        
        team_stats[team] = {
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'total_games': total_games
        }
    
    return team_stats
    """Make prediction for a match"""
    try:
        # Encode team names
        if local_team not in local_encoder.classes_:
            st.warning(f"Warning: {local_team} not in training data. Using default encoding.")
            local_encoded = 0
        else:
            local_encoded = local_encoder.transform([local_team])[0]
            
        if visitor_team not in visitor_encoder.classes_:
            st.warning(f"Warning: {visitor_team} not in training data. Using default encoding.")
            visitor_encoded = 0
        else:
            visitor_encoded = visitor_encoder.transform([visitor_team])[0]
        
        # Encode other features
        season_encoded = season_encoder.transform([season])[0] if season in season_encoder.classes_ else 0
        division_encoded = division_encoder.transform([division])[0] if division in division_encoder.classes_ else 0
        round_encoded = round_encoder.transform([round_val])[0] if round_val in round_encoder.classes_ else 0
        
        # Get current date features
        current_date = datetime.now()
        month = current_date.month
        day_of_week = current_date.weekday()
        
        # Get team statistics
        local_win_rate = team_stats.get(local_team, {}).get('win_rate', 0.5)
        visitor_win_rate = team_stats.get(visitor_team, {}).get('win_rate', 0.5)
        local_draw_rate = team_stats.get(local_team, {}).get('draw_rate', 0.3)
        visitor_draw_rate = team_stats.get(visitor_team, {}).get('draw_rate', 0.3)
        
        # Create feature array
        features = np.array([[
            local_encoded, visitor_encoded, season_encoded, division_encoded, 
            round_encoded, month, day_of_week, local_win_rate, visitor_win_rate,
            local_draw_rate, visitor_draw_rate
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get class names
        classes = model.classes_
        prob_dict = {classes[i]: probabilities[i] for i in range(len(classes))}
        
        return prediction, prob_dict
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models and data..."):
        model, local_encoder, visitor_encoder, season_encoder, division_encoder, round_encoder, team_stats = load_models_and_data()
    
    if model is None:
        st.stop()
    
    st.success("Models loaded successfully! 🎉")
    
    # Sidebar for inputs
    st.sidebar.header("Match Details")
    
    # Get unique teams from encoders
    local_teams = list(local_encoder.classes_)
    visitor_teams = list(visitor_encoder.classes_)
    seasons = list(season_encoder.classes_)
    divisions = list(division_encoder.classes_)
    rounds = list(round_encoder.classes_)
    
    # Team selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        local_team = st.selectbox("Select Home Team", local_teams, key="local")
        
        # Show team stats if available
        if local_team in team_stats:
            stats = team_stats[local_team]
            st.metric("Win Rate", f"{stats['win_rate']:.2%}")
            st.metric("Total Games", stats['total_games'])
    
    with col2:
        st.subheader("✈️ Away Team")
        visitor_team = st.selectbox("Select Away Team", visitor_teams, key="visitor")
        
        # Show team stats if available
        if visitor_team in team_stats:
            stats = team_stats[visitor_team]
            st.metric("Win Rate", f"{stats['win_rate']:.2%}")
            st.metric("Total Games", stats['total_games'])
    
    # Additional match details
    st.sidebar.subheader("Match Context")
    season = st.sidebar.selectbox("Season", seasons)
    division = st.sidebar.selectbox("Division", divisions)
    round_val = st.sidebar.selectbox("Round", rounds)
    
    # Prediction button
    if st.button("🔮 Predict Match Result", type="primary", use_container_width=True):
        if local_team == visitor_team:
            st.error("Please select different teams for home and away!")
        else:
            with st.spinner("Making prediction..."):
                prediction, probabilities = predict_match(
                    model, local_team, visitor_team, season, division, round_val,
                    local_encoder, visitor_encoder, season_encoder, division_encoder,
                    round_encoder, team_stats
                )
            
            if prediction is not None:
                st.subheader("🎯 Prediction Results")
                
                # Display main prediction
                if prediction == "Local Win":
                    st.markdown(f'''
                    <div class="prediction-box win-box">
                        <h2>🏆 Predicted Winner: {local_team}</h2>
                        <p>The home team is predicted to win this match!</p>
                    </div>
                    ''', unsafe_allow_html=True)
                elif prediction == "Visitor Win":
                    st.markdown(f'''
                    <div class="prediction-box win-box">
                        <h2>🏆 Predicted Winner: {visitor_team}</h2>
                        <p>The away team is predicted to win this match!</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-box draw-box">
                        <h2>🤝 Predicted Result: Draw</h2>
                        <p>This match is predicted to end in a draw!</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Display probabilities
                st.subheader("📊 Prediction Probabilities")
                prob_df = pd.DataFrame(list(probabilities.items()), columns=['Outcome', 'Probability'])
                prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                prob_df = prob_df.sort_values('Probability', ascending=False)
                
                col1, col2, col3 = st.columns(3)
                
                for i, (outcome, prob) in enumerate(probabilities.items()):
                    with [col1, col2, col3][i % 3]:
                        st.metric(
                            outcome.replace('Local Win', f'{local_team} Win').replace('Visitor Win', f'{visitor_team} Win'),
                            f"{prob:.2%}"
                        )
                
                # Show detailed breakdown
                with st.expander("📈 Detailed Analysis"):
                    st.write("**Match Context:**")
                    st.write(f"- **Season:** {season}")
                    st.write(f"- **Division:** {division}")
                    st.write(f"- **Round:** {round_val}")
                    st.write(f"- **Date:** {datetime.now().strftime('%Y-%m-%d')}")
                    
                    if local_team in team_stats and visitor_team in team_stats:
                        local_stats = team_stats[local_team]
                        visitor_stats = team_stats[visitor_team]
                        
                        st.write("**Historical Performance:**")
                        comparison_df = pd.DataFrame({
                            'Metric': ['Win Rate', 'Draw Rate', 'Total Games'],
                            local_team: [f"{local_stats['win_rate']:.2%}", 
                                       f"{local_stats['draw_rate']:.2%}", 
                                       local_stats['total_games']],
                            visitor_team: [f"{visitor_stats['win_rate']:.2%}", 
                                         f"{visitor_stats['draw_rate']:.2%}", 
                                         visitor_stats['total_games']]
                        })
                        st.dataframe(comparison_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("**Note:** Predictions are based on historical data and machine learning models. Actual match results may vary!")
    
    # Model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Model Information")
    st.sidebar.write("**Algorithm:** Random Forest")
    st.sidebar.write("**Features Used:**")
    st.sidebar.write("- Team identities")
    st.sidebar.write("- Season and division")
    st.sidebar.write("- Match round")
    st.sidebar.write("- Historical win/draw rates")
    st.sidebar.write("- Date features")

if __name__ == "__main__":
    main()