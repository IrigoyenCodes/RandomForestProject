from randomForest2 import NBAPredictorModel

if __name__ == "__main__":
    # Initialize and train model
    predictor = NBAPredictorModel()
    
    if predictor.fetch_team_stats():
        try:
            predictor.train_model()
            print("Model trained successfully!")
            print("Generated files:")
            print("- nba_rf_model.pkl")
            print("- team_stats.csv")
        except Exception as e:
            print(f"Training failed: {str(e)}")
    else:
        print("Failed to fetch team stats")