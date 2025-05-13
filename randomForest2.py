import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import requests
import os

class NBAPredictorModel:
    def __init__(self):
        self.model = None
        self.team_stats = None
        self.features = [
            'PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 
            'FG_PCT', 'FG3_PCT', 'FT_PCT', 'W_PCT'
        ]
        
    def fetch_team_stats(self, season='2023-24'):
        """Fetch current team stats from NBA API"""
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://www.nba.com/'
        }
        
        url = f"https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Create DataFrame from API response
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            self.team_stats = pd.DataFrame(rows, columns=headers)
            
            # Set team name as index for easier lookup
            self.team_stats.set_index('TEAM_NAME', inplace=True)
            
            return True
        except Exception as e:
            print(f"Error fetching team stats: {e}")
            return False
    
    def train_model(self, save_path='nba_rf_model.pkl'):
        """Train the RandomForest model using current team stats"""
        if self.team_stats is None:
            if not self.fetch_team_stats():
                raise ValueError("Could not fetch team stats for training")
        
        # Generate all possible matchups
        teams = self.team_stats.index.tolist()
        matchups = []
        
        for home in teams:
            for away in teams:
                if home != away:
                    # Calculate feature differences
                    features = {}
                    for feat in self.features:
                        features[f'{feat}_diff'] = (
                            self.team_stats.loc[home, feat] - 
                            self.team_stats.loc[away, feat]
                        )
                    
                    # Simulate outcome (in real app, use historical data)
                    home_win = (
                        self.team_stats.loc[home, 'W_PCT'] - 
                        self.team_stats.loc[away, 'W_PCT'] + 
                        0.05  # home advantage
                    ) > 0
                    
                    features['outcome'] = 1 if home_win else 0
                    matchups.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(matchups)
        X = df.drop('outcome', axis=1)
        y = df['outcome']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=5,
            min_samples_split=5
        )
        self.model.fit(X_train, y_train)
        
        # Save model and stats
        self.save_model(save_path)
        self.team_stats.to_csv('team_stats.csv')
        
        return self.model
    
    def save_model(self, path):
        """Save trained model to file"""
        if self.model:
            joblib.dump(self.model, path)
            print(f"Model saved to {path}")
    
    def load_model(self, model_path='nba_rf_model.pkl', stats_path='team_stats.csv'):
        """Load trained model and team stats"""
        if not os.path.exists(model_path) or not os.path.exists(stats_path):
            raise FileNotFoundError("Model or stats file not found")
            
        self.model = joblib.load(model_path)
        self.team_stats = pd.read_csv(stats_path, index_col='TEAM_NAME')
        return self.model, self.team_stats
    
    def predict_winner(self, team1, team2):
        """Predict winner between two teams"""
        if team1 not in self.team_stats.index:
            raise ValueError(f"Team {team1} not found in stats")
        if team2 not in self.team_stats.index:
            raise ValueError(f"Team {team2} not found in stats")
        
        # Calculate feature differences
        input_features = []
        for feat in self.features:
            diff = (
                self.team_stats.loc[team1, feat] - 
                self.team_stats.loc[team2, feat]
            )
            input_features.append(diff)
        
        # Make prediction
        proba = self.model.predict_proba([input_features])[0]
        prediction = self.model.predict([input_features])[0]
        
        # Determine winner and confidence
        winner = team1 if prediction == 1 else team2
        confidence = "high" if max(proba) > 0.7 else "medium" if max(proba) > 0.6 else "low"
        
        return {
            'team1': team1,
            'team2': team2,
            'winner': winner,
            'confidence': confidence,
            'team1_win_prob': float(proba[1]),
            'team2_win_prob': float(proba[0]),
            'features': dict(zip(self.features, input_features))
        }

# Integration functions for your GUI
def load_model_and_teams():
    predictor = NBAPredictorModel()
    model, team_stats = predictor.load_model()
    return model, team_stats

def predict_winner(model, team_stats, team1, team2):
    predictor = NBAPredictorModel()
    predictor.model = model
    predictor.team_stats = team_stats
    result = predictor.predict_winner(team1, team2)
    return result['winner'], max(result['team1_win_prob'], result['team2_win_prob'])