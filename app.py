import streamlit as st
import pickle
import pandas as pd
import numpy as np

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title('IPL Win Predictor')

# FIX 1: Use st.columns instead of st.beta_columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

# FIX 2: Add min_value for logical input
target = st.number_input('Target', min_value=1, step=1)

# FIX 3: Change layout for Overs AND Balls. Use st.columns.
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, step=1)
with col4:
    # Use two inputs for 'Overs' and 'Balls' for accuracy
    overs = st.number_input('Overs', min_value=0, max_value=19, step=1)
with col5:
    balls = st.number_input('Balls', min_value=0, max_value=5, step=1)

# FIX 4: Rename 'wickets' to 'wickets_out' for clarity
wickets_out = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

if st.button('Predict Probability'):
    
    # FIX 5: Add validation logic
    if batting_team == bowling_team:
        st.error('Batting team and bowling team must be different.')
    else:
        # FIX 6: Correct calculation for balls_left and total_overs
        runs_left = target - score
        balls_bowled = (overs * 6) + balls
        balls_left = 120 - balls_bowled
        wickets_left = 10 - wickets_out
        total_overs = overs + (balls / 6)

        # FIX 7: Handle ZeroDivisionError for CRR
        if total_overs == 0:
            crr = 0
        else:
            crr = score / total_overs

        # FIX 7: Handle ZeroDivisionError for RRR
        if balls_left == 0:
            if runs_left > 0:
                # Team lost, RRR is effectively infinite
                rrr = 999
            else:
                # Team won or tied, RRR is 0
                rrr = 0
        else:
            rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],  # Use the calculated wickets_left
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.header(f"{batting_team} - {round(win * 100)}%")
        st.header(f"{bowling_team} - {round(loss * 100)}%")