import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
import os
import joblib
from randomForest2 import NBAPredictorModel  # Ensure original code is in 'nba_predictor.py'

def plot_feature_importance(model, feature_names, n=10):
    """Plot feature importance horizontal bar chart"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-n:]
    
    plt.figure(figsize=(10, 6))
    plt.title('Top {} Feature Importances'.format(n))
    plt.barh(range(n), importances[indices], align='center')
    plt.yticks(range(n), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

def main():
    # Initialize styling
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    
    predictor = NBAPredictorModel()
    model_path = 'nba_rf_model.pkl'
    stats_path = 'team_stats.csv'

    # Train new model if not found
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print("Model not found. Training new model...")
        if not predictor.fetch_team_stats():
            print("Failed to fetch team stats. Exiting.")
            return
        predictor.train_model()
        print("Model trained and saved.")

    # Load model and stats
    model, team_stats = predictor.load_model()
    predictor.team_stats = team_stats

    # Generate evaluation dataset
    features = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 
                'FG_PCT', 'FG3_PCT', 'FT_PCT', 'W_PCT']
    teams = team_stats.index.tolist()
    matchups = []

    for home in teams:
        for away in teams:
            if home != away:
                feat_dict = {}
                for feat in features:
                    feat_diff = team_stats.loc[home, feat] - team_stats.loc[away, feat]
                    feat_dict[f'{feat}_diff'] = feat_diff
                home_win_prob = (team_stats.loc[home, 'W_PCT'] - 
                                team_stats.loc[away, 'W_PCT'] + 0.05)
                feat_dict['outcome'] = 1 if home_win_prob > 0 else 0
                matchups.append(feat_dict)

    df = pd.DataFrame(matchups)
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Detailed Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))

    print("\n=== Key Performance Metrics ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")

    # Confusion Matrix Visualization
    plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Away Win', 'Home Win'], 
               yticklabels=['Away Win', 'Home Win'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.show()

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title('Precision-Recall Curve')
    plt.show()

    # Feature Importance
    plot_feature_importance(model, X.columns.tolist(), n=10)

    # Interactive prediction section remains the same...
    
    # Interactive team selection
    print("\n=== Team Selection ===")
    print("Available Teams:")
    for idx, team in enumerate(teams, 1):
        print(f"{idx:2}. {team}")

    while True:
        try:
            team1_idx = int(input("\nEnter HOME team number: ")) - 1
            team2_idx = int(input("Enter AWAY team number: ")) - 1
            
            if team1_idx < 0 or team2_idx < 0 or team1_idx >= len(teams) or team2_idx >= len(teams):
                print("Invalid team numbers. Try again.")
                continue
                
            team1 = teams[team1_idx]
            team2 = teams[team2_idx]
            
            if team1 == team2:
                print("Teams must be different. Try again.")
                continue
                
            break
        except ValueError:
            print("Please enter valid numbers.")

    # Get prediction
    result = predictor.predict_winner(team1, team2)

    print("\n=== Prediction Result ===")
    print(f"Home Team: {result['team1']}")
    print(f"Away Team: {result['team2']}")
    print(f"Winner:    {result['winner']} (confidence: {result['confidence']*100:.1f}%)")
    print(f"Probabilities: {result['team1']} {result['team1_win_prob']*100:.1f}% - " 
          f"{result['team2']} {result['team2_win_prob']*100:.1f}%")


if __name__ == "__main__":
    main()