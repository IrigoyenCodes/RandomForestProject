from multiprocessing import process
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from scipy import stats
import os
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

class NBAPredictorModel:
    def __init__(self, season="2024-25"):
        self.season = season
        self.player_stats = None
        self.rf_model = None
        self.feature_names = None
        self.model_trained = False
    
    def fetch_data(self):
        """Fetch NBA player game logs from the NBA API"""
        print(Fore.YELLOW + "\nFetching data from NBA API..." + Style.RESET_ALL)
        
        url = 'https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=4&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2024-25&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight='
        
        params = {
        'College': '',
        'Conference': '',
        'Country': '',
        'DateFrom': '',
        'DateTo': '',
        'Division': '',
        'DraftPick': '',
        'DraftYear': '',
        'GameScope': '',
        'GameSegment': '',
        'Height': '',
        'ISTRound': '',
        'LastNGames': '4',
        'LeagueID': '00',
        'Location': '',
        'MeasureType': 'Base',
        'Month': '0',
        'OpponentTeamID': '0',
        'Outcome': '',
        'PORound': '0',
        'PaceAdjust': 'N',
        'PerMode': 'PerGame',
        'Period': '0',
        'PlayerExperience': '',
        'PlayerPosition': '',
        'PlusMinus': 'N',
        'Rank': 'N',
        'Season': '2024-25',
        'SeasonSegment': '',
        'SeasonType': 'Regular Season',
        'ShotClockRange': '',
        'StarterBench': '',
        'TeamID': '0',
        'VsConference': '',
        'VsDivision': '',
        'Weight': ''
    }

        
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'Accept': 'application/json',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true'
    }
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            
            self.player_stats = pd.DataFrame(rows, columns=headers)
            print(Fore.GREEN + "Data loaded successfully!" + Style.RESET_ALL)
            return True
            
        except requests.exceptions.RequestException as e:
            print(Fore.RED + f"Error fetching data: {e}" + Style.RESET_ALL)
            return False
        except (KeyError, ValueError) as e:
            print(Fore.RED + f"Error processing data: {e}" + Style.RESET_ALL)
            return False
    


    def _get_player_stats(self, player_name_or_id):
        """Get player stats by name or ID"""
        # Check cache first
        if player_name_or_id in self.player_cache:
            return self.player_cache[player_name_or_id]
        
        # Try to find player by ID
        if isinstance(player_name_or_id, int) or player_name_or_id.isdigit():
            player_id = int(player_name_or_id)
            player_data = self.player_stats[self.player_stats['PLAYER_ID'] == player_id]
            if not player_data.empty:
                self.player_cache[player_name_or_id] = player_data.iloc[0]
                return player_data.iloc[0]
        
        # Try to find player by name (fuzzy matching)
        if self.all_players:
            matches = process.extractOne(player_name_or_id, self.all_players.keys())
            if matches and matches[1] > 80:  # Only accept good matches
                player_name = matches[0]
                player_id = self.all_players[player_name]
                player_data = self.player_stats[self.player_stats['PLAYER_ID'] == player_id]
                if not player_data.empty:
                    self.player_cache[player_name_or_id] = player_data.iloc[0]
                    return player_data.iloc[0]
        
        print(Fore.RED + f"Player '{player_name_or_id}' not found." + Style.RESET_ALL)
        return None
    
    def interactive_predict(self):
        """Make predictions by entering just player name or ID"""
        if not self.model_trained:
            print(Fore.RED + "Error: Model not trained yet." + Style.RESET_ALL)
            return
        
        if self.player_stats is None:
            print(Fore.RED + "Error: No player data loaded. Fetch data first." + Style.RESET_ALL)
            return
        
        while True:
            print(Fore.YELLOW + "\nEnter player name or ID (or 'back' to return):" + Style.RESET_ALL)
            player_input = input(Fore.CYAN + "Player: " + Style.RESET_ALL)
            
            if player_input.lower() == 'back':
                break
            
            player_data = self._get_player_stats(player_input)
            if player_data is None:
                continue
            
            # Prepare input data
            input_data = {}
            missing_features = []
            
            for feature in self.feature_names:
                if feature in player_data:
                    input_data[feature] = player_data[feature]
                else:
                    missing_features.append(feature)
            
            if missing_features:
                print(Fore.RED + f"\nWarning: Missing features - {', '.join(missing_features)}" + Style.RESET_ALL)
                print("Please provide these values:")
                
                for feature in missing_features:
                    while True:
                        try:
                            value = float(input(Fore.CYAN + f"{feature}: " + Style.RESET_ALL))
                            input_data[feature] = value
                            break
                        except ValueError:
                            print(Fore.RED + "Please enter a valid number." + Style.RESET_ALL)
            
            # Make prediction
            input_df = pd.DataFrame([input_data])
            prediction = self.rf_model.predict(input_df)
            
            print(Fore.GREEN + f"\nPrediction for {player_data['PLAYER_NAME'] if 'PLAYER_NAME' in player_data else player_input}:" + Style.RESET_ALL)
            print(f"Predicted {self.target_variable}: {prediction[0]:.2f}")
            
            # Show actual stats if available
            if self.target_variable in player_data:
                print(f"Actual {self.target_variable}: {player_data[self.target_variable]:.2f}")
                print(f"Difference: {abs(player_data[self.target_variable] - prediction[0]):.2f}")
            
            print("\n" + "="*50)



    def load_data_from_csv(self):
        """Load player data from a CSV file"""
        while True:
            file_path = input(Fore.CYAN + "Enter path to CSV file (or 'back' to return): " + Style.RESET_ALL)
            if file_path.lower() == 'back':
                return False
            
            try:
                self.player_stats = pd.read_csv(file_path)
                print(Fore.GREEN + f"Data loaded successfully from {file_path}!" + Style.RESET_ALL)
                return True
            except FileNotFoundError:
                print(Fore.RED + f"Error: File not found at {file_path}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"Error loading file: {e}" + Style.RESET_ALL)
    
    def prepare_data(self):
        """Prepare data for modeling with interactive options"""
        if self.player_stats is None:
            print(Fore.RED + "Error: No data loaded. Please fetch data first." + Style.RESET_ALL)
            return False
        
        # Show data preview
        print(Fore.YELLOW + "\nData Preview:" + Style.RESET_ALL)
        print(self.player_stats.head())
        print(Fore.YELLOW + "\nAvailable Columns:" + Style.RESET_ALL)
        print(self.player_stats.columns.tolist())
        
        # Select target variable
        while True:
            self.target_variable = input(Fore.CYAN + "\nEnter target variable to predict (e.g., PTS, AST, REB): " + Style.RESET_ALL)
            if self.target_variable in self.player_stats.columns:
                break
            print(Fore.RED + f"Error: '{self.target_variable}' not found in data. Try again." + Style.RESET_ALL)
        
        print(Fore.GREEN + f"\nTarget variable set to: {self.target_variable}" + Style.RESET_ALL)
        
        # Feature engineering
        print(Fore.YELLOW + "\nFeature Engineering Options:" + Style.RESET_ALL)
        if 'FGM' in self.player_stats.columns and 'FGA' in self.player_stats.columns:
            self.player_stats['FG_PCT'] = np.where(
                self.player_stats['FGA'] > 0,
                self.player_stats['FGM'] / self.player_stats['FGA'],
                self.player_stats['FGM'].median()
            )
            print("- Added FG_PCT (Field Goal Percentage)")
        
        if 'FG3M' in self.player_stats.columns and 'FG3A' in self.player_stats.columns:
            self.player_stats['FG3_PCT'] = np.where(
                self.player_stats['FG3A'] > 0,
                self.player_stats['FG3M'] / self.player_stats['FG3A'],
                self.player_stats['FG3M'].median()
            )
            print("- Added FG3_PCT (3-Point Percentage)")
        
        if 'FTM' in self.player_stats.columns and 'FTA' in self.player_stats.columns:
            self.player_stats['FT_PCT'] = np.where(
                self.player_stats['FTA'] > 0,
                self.player_stats['FTM'] / self.player_stats['FTA'],
                self.player_stats['FTM'].median()
            )
            print("- Added FT_PCT (Free Throw Percentage)")
        
        # Select features interactively
        print(Fore.YELLOW + "\nAvailable Features:" + Style.RESET_ALL)
        available_features = [col for col in self.player_stats.columns if col != self.target_variable]
        for i, feat in enumerate(available_features, 1):
            print(f"{i}. {feat}")
        
        print(Fore.YELLOW + "\nSelect features to include (comma-separated numbers, or 'all'):" + Style.RESET_ALL)
        while True:
            selection = input(Fore.CYAN + "Your selection: " + Style.RESET_ALL)
            if selection.lower() == 'all':
                self.player_features = available_features
                break
            try:
                selected_indices = [int(i.strip()) - 1 for i in selection.split(',')]
                self.player_features = [available_features[i] for i in selected_indices]
                break
            except (ValueError, IndexError):
                print(Fore.RED + "Invalid selection. Please try again." + Style.RESET_ALL)
        
        print(Fore.GREEN + f"\nSelected features: {', '.join(self.player_features)}" + Style.RESET_ALL)
        
        # Handle missing values
        print(Fore.YELLOW + "\nMissing Value Report:" + Style.RESET_ALL)
        print(self.player_stats[self.player_features].isnull().sum())
        
        for col in self.player_features:
            if self.player_stats[col].isnull().sum() > 0:
                print(Fore.YELLOW + f"\nHandling missing values for {col}:" + Style.RESET_ALL)
                print("1. Fill with mean")
                print("2. Fill with median")
                print("3. Fill with zero")
                print("4. Drop rows with missing values")
                
                choice = input(Fore.CYAN + "Select method (1-4): " + Style.RESET_ALL)
                if choice == '1':
                    self.player_stats[col].fillna(self.player_stats[col].mean(), inplace=True)
                elif choice == '2':
                    self.player_stats[col].fillna(self.player_stats[col].median(), inplace=True)
                elif choice == '3':
                    self.player_stats[col].fillna(0, inplace=True)
                elif choice == '4':
                    self.player_stats.dropna(subset=[col], inplace=True)
        
        return True
    
    def train_model(self):
        """Train the model with interactive options"""
        if not hasattr(self, 'player_features') or not hasattr(self, 'target_variable'):
            print(Fore.RED + "Error: Data not prepared. Run prepare_data() first." + Style.RESET_ALL)
            return False
        
        # Get user input for test size
        while True:
            test_size = input(Fore.CYAN + "\nEnter test size proportion (0.1-0.5, default 0.2): " + Style.RESET_ALL)
            if not test_size:
                test_size = 0.2
                break
            try:
                test_size = float(test_size)
                if 0.1 <= test_size <= 0.5:
                    break
                print(Fore.RED + "Please enter a value between 0.1 and 0.5" + Style.RESET_ALL)
            except ValueError:
                print(Fore.RED + "Invalid input. Please enter a number." + Style.RESET_ALL)
        
        # Prepare data
        X = self.player_stats[self.player_features]
        y = self.player_stats[self.target_variable]
        
        # Convert categorical variables if needed
        categorical_cols = X.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            print(Fore.YELLOW + f"\nFound categorical columns: {', '.join(categorical_cols)}" + Style.RESET_ALL)
            print("1. One-hot encode (create dummy variables)")
            print("2. Drop categorical columns")
            
            choice = input(Fore.CYAN + "Select method (1-2): " + Style.RESET_ALL)
            if choice == '1':
                X = pd.get_dummies(X, columns=categorical_cols)
            else:
                X = X.drop(columns=categorical_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.test_indices = X_test.index
        self.feature_names = X.columns.tolist()
        
        print(Fore.GREEN + f"\nTraining set: {X_train.shape[0]} samples" + Style.RESET_ALL)
        print(Fore.GREEN + f"Test set: {X_test.shape[0]} samples" + Style.RESET_ALL)
        
        # Model configuration
        print(Fore.YELLOW + "\nModel Configuration:" + Style.RESET_ALL)
        print("1. Default Random Forest (100 trees)")
        print("2. Custom Configuration")
        
        choice = input(Fore.CYAN + "Select option (1-2): " + Style.RESET_ALL)
        if choice == '1':
            self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            n_estimators = int(input("Number of trees (default 100): ") or 100)
            max_depth = input("Max tree depth (default None, or enter number): ")
            max_depth = None if max_depth == '' else int(max_depth)
            
            self.rf_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        
        # Train model
        print(Fore.YELLOW + "\nTraining model..." + Style.RESET_ALL)
        self.rf_model.fit(X_train, y_train)
        self.model_trained = True
        
        # Evaluate
        y_pred = self.rf_model.predict(X_test)
        self.y_test = y_test
        self.y_pred = y_pred
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(Fore.GREEN + "\nModel Evaluation:" + Style.RESET_ALL)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        
        return True
    
    def visualize_results(self):
        """Interactive visualization menu"""
        if not self.model_trained:
            print(Fore.RED + "Error: Model not trained yet." + Style.RESET_ALL)
            return
        
        while True:
            print(Fore.YELLOW + "\nVisualization Options:" + Style.RESET_ALL)
            print("1. Predictions vs Actual Values")
            print("2. Feature Importance")
            print("3. Error Distribution")
            print("4. All Visualizations")
            print("5. Return to Main Menu")
            
            choice = input(Fore.CYAN + "Select option (1-5): " + Style.RESET_ALL)
            
            if choice == '5':
                break
            
            plt.figure(figsize=(10, 6))
            
            if choice in ['1', '4']:
                plt.scatter(self.y_test, self.y_pred, alpha=0.7)
                plt.plot([self.y_test.min(), self.y_test.max()], 
                         [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predictions')
                plt.title(f'Predictions vs Actual {self.target_variable}')
                plt.grid(True)
                plt.show()
            
            if choice in ['2', '4']:
                importances = self.rf_model.feature_importances_
                feat_imp = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                plt.barh(feat_imp['Feature'], feat_imp['Importance'])
                plt.xlabel("Importance")
                plt.title("Feature Importance")
                plt.gca().invert_yaxis()
                plt.show()
            
            if choice in ['3', '4']:
                errors = self.y_test - self.y_pred
                plt.hist(errors, bins=30, alpha=0.7)
                plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
                plt.xlabel("Prediction Error")
                plt.ylabel("Frequency")
                plt.title("Error Distribution")
                plt.grid(True, alpha=0.3)
                plt.show()
    
    def save_model(self):
        """Save the trained model to disk"""
        if not self.model_trained:
            print(Fore.RED + "Error: No trained model to save." + Style.RESET_ALL)
            return
        
        model_name = input(Fore.CYAN + "Enter name for model (without extension): " + Style.RESET_ALL)
        if not model_name:
            model_name = "nba_predictor"
        
        try:
            # Save model
            with open(f"{model_name}.pkl", 'wb') as f:
                pickle.dump(self.rf_model, f)
            
            # Save feature names
            with open(f"{model_name}_features.txt", 'w') as f:
                f.write(','.join(self.feature_names))
            
            print(Fore.GREEN + f"Model saved as {model_name}.pkl" + Style.RESET_ALL)
            print(Fore.GREEN + f"Feature names saved as {model_name}_features.txt" + Style.RESET_ALL)
            return True
        except Exception as e:
            print(Fore.RED + f"Error saving model: {e}" + Style.RESET_ALL)
            return False
    
    def load_model(self):
        """Load a trained model from disk"""
        model_file = input(Fore.CYAN + "Enter model filename (.pkl): " + Style.RESET_ALL)
        features_file = input(Fore.CYAN + "Enter feature names filename (.txt): " + Style.RESET_ALL)
        
        try:
            with open(model_file, 'rb') as f:
                self.rf_model = pickle.load(f)
            
            with open(features_file, 'r') as f:
                self.feature_names = f.read().split(',')
            
            self.model_trained = True
            print(Fore.GREEN + "Model loaded successfully!" + Style.RESET_ALL)
            return True
        except FileNotFoundError:
            print(Fore.RED + "Error: File not found." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error loading model: {e}" + Style.RESET_ALL)
        return False
    
    def interactive_predict(self):
        """Make predictions on new data interactively"""
        if not self.model_trained:
            print(Fore.RED + "Error: Model not trained yet." + Style.RESET_ALL)
            return
        
        print(Fore.YELLOW + "\nEnter player stats for prediction:" + Style.RESET_ALL)
        input_data = {}
        for feature in self.feature_names:
            while True:
                value = input(Fore.CYAN + f"{feature}: " + Style.RESET_ALL)
                try:
                    input_data[feature] = float(value)
                    break
                except ValueError:
                    print(Fore.RED + "Please enter a valid number." + Style.RESET_ALL)
        
        input_df = pd.DataFrame([input_data])
        prediction = self.rf_model.predict(input_df)
        
        print(Fore.GREEN + f"\nPredicted {self.target_variable}: {prediction[0]:.2f}" + Style.RESET_ALL)

def main_menu():
    """Main interactive menu"""
    model = NBAPredictorModel()
    
    while True:
        print(Fore.BLUE + "\nNBA Player Performance Predictor" + Style.RESET_ALL)
        print("=" * 40)
        print("1. Load Data (from API)")
        print("2. Load Data (from CSV)")
        print("3. Prepare Data")
        print("4. Train Model")
        print("5. Visualize Results")
        print("6. Save Model")
        print("7. Load Model")
        print("8. Make Prediction")
        print("9. Exit")
        
        choice = input(Fore.CYAN + "\nSelect option (1-9): " + Style.RESET_ALL)
        
        if choice == '1':
            model.fetch_data()
        elif choice == '2':
            model.load_data_from_csv()
        elif choice == '3':
            model.prepare_data()
        elif choice == '4':
            model.train_model()
        elif choice == '5':
            model.visualize_results()
        elif choice == '6':
            model.save_model()
        elif choice == '7':
            model.load_model()
        elif choice == '8':
            if model.model_trained:
                model.interactive_predict()
            else:
                print(Fore.RED + "Please train or load a model first." + Style.RESET_ALL)
        elif choice == '9':
            print(Fore.YELLOW + "Exiting..." + Style.RESET_ALL)
            break
        else:
            print(Fore.RED + "Invalid choice. Please try again." + Style.RESET_ALL)

if __name__ == "__main__":
    main_menu()