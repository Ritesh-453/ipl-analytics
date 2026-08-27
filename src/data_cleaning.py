# notebooks/member1/01_data_cleaning.ipynb
import pandas as pd

matches = pd.read_csv('data/raw/matches.csv')
print(matches.shape)
print(matches.isnull().sum())

# Fix common issues
matches['winner'].fillna('No Result', inplace=True)
matches['player_of_match'].fillna('Unknown', inplace=True)

# Standardize team names (they change over years)
team_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
}
matches['team1'] = matches['team1'].replace(team_map)
matches['team2'] = matches['team2'].replace(team_map)
matches['winner'] = matches['winner'].replace(team_map)

matches.to_csv('data/processed/matches_clean.csv', index=False)
print('Cleaned matches saved.')
