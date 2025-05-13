# RandomForestProject
RandomForest made for AI course, includes a GUI for the DeepLearning algorithm 
## Overview

The NBA Game Predictor is a Python application that uses machine learning (Random Forest algorithm) to predict the outcome of NBA games based on team statistics. The application features a graphical user interface built with Tkinter and ttkbootstrap for a modern look.

## Key Features

1. **Game Prediction**: Predicts winners between two selected NBA teams with confidence percentage
    
2. **Team Statistics Comparison**: Displays key stats comparison between selected teams
    
3. **NBA Playoff Results**: Shows recent playoff game results and series standings
    
4. **Player Standings**: Displays player statistics leaders for the current season
    
5. **Team Information**: Provides a list of all NBA teams
    

## File Structure

- `randomForest2.py`: Contains the machine learning model for predictions
    
- `GUiEXample.py`: Main application file with GUI implementation, includes randomForest2 
    
- `test_import.py`: Import Rainforest.py into the main file and check if the import was successfull
     
- `train_model.py`:Initialize and train model
    
## Classes

### 1. GUiEXample

The main application class that creates the GUI and handles all functionality.

#### Methods:

- `__init__()`: Initializes the application window and components
    
- `create_navbar()`: Creates the top navigation bar
    
- `create_main_content()`: Creates the main content area with team selection and results display
    
- `make_prediction()`: Makes a prediction between two selected teams
    
- `load_playoff_data()`: Loads NBA playoff data in a background thread
    
- `fetch_nba_playoff_gamelog()`: Fetches playoff game log from NBA API
    
- `process_game_log_data()`: Processes raw game log data into a DataFrame
    
- `display_game_log_in_ui()`: Displays game log in the results text widget
    
- `display_series_tracker_in_ui()`: Displays playoff series standings
    
- `show_teams_window()`: Shows the teams list window
    
- `show_player_standings_window()`: Shows the player standings window
    

### 2. TeamsWindow

Displays a simple list of all NBA teams.

### 3. SimplePlayerStandings

Displays player statistics leaders with filtering options.

### 4. TeamsWindow2

(Note: This class appears to be incomplete in the provided code)

## Dependencies

- Python 3.x
    
- Required packages:
    
    - tkinter
        
    - ttkbootstrap
        
    - pandas
        
    - requests
        
    - threading
        

## Usage

1. Run the application with `python nba_predictor.py`
    
2. Select two teams from the dropdown menus
    
3. Click "Predict Winner" to see the prediction and stats comparison
    
4. Use the navigation buttons to access additional features:
    
    - "NBA TEAMS" - Shows list of all teams
        
    - "NBA PLAYER STANDINGS" - Shows player statistics leaders
        
    - "GENERAL STATS" - (Functionality not fully implemented)
        

## Data Sources

The application fetches real-time data from:

- NBA API ([https://stats.nba.com](https://stats.nba.com/)) for playoff results and player statistics
    

## Styling

The application uses ttkbootstrap with the "solar" theme for a modern, consistent look across all components.

## Error Handling

The application includes basic error handling for:

- API request failures
    
- Missing data
    
- UI update threading issues
    

## Limitations

1. Requires internet connection to fetch latest data
    
2. Machine learning model accuracy depends on the training data
    
3. Some features may not be fully implemented (like GENERAL STATS)
    

## Future Enhancements

1. Add more detailed team statistics
    
2. Implement historical data analysis
    
3. Add visualization of prediction results
    
4. Improve error handling and user feedback
    
5. Add caching for API responses
    
6. Improve ML/DeepLearning algorithm
