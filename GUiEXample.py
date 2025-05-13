import sys
if 'randomForest2' in sys.modules:
    del sys.modules['randomForest2']

from randomForest2 import NBAPredictorModel
print(f"Imported NBAPredictorModel from: {NBAPredictorModel.__module__}")

import os
rf_path = os.path.abspath('randomForest2.py')
print(f"Attempting to import from: {rf_path}")
print(f"File exists: {os.path.exists(rf_path)}")

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import requests
import pandas as pd
from datetime import datetime
import threading
import randomForest2

class NBAPredictor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NBA Game Predictor")
        self.geometry("1350x800")

        # Configure theme
        self.style = tb.Style(theme="solar")
        
        # NBA teams (for demonstration)
        self.nba_teams = [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
            "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
            "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
            "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
            "Utah Jazz", "Washington Wizards"
        ]
        
        # Create main frame
        self.main_frame = tb.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top navigation bar
        self.create_navbar()
        
        # Create main content
        self.create_main_content()
        
        # Load NBA playoff data
        self.playoff_data = None
        self.processed_data = None
        threading.Thread(target=self.load_playoff_data).start()

        self.configure_text_tags()  # Add this line

    def make_prediction(self):
        team1 = self.team1_var.get()
        team2 = self.team2_var.get()
        
        model, team_stats = randomForest2.load_model_and_teams()
        winner, confidence = randomForest2.predict_winner(model, team_stats, team1, team2)
        
        # Get team stats
        t1_stats = team_stats.loc[team1]
        t2_stats = team_stats.loc[team2]
        

        # Build output
        self.rf_output_text.config(state="normal")
        self.rf_output_text.delete(1.0, tk.END)
        
        # Header
        self.rf_output_text.insert(tk.END, "PREDICTION RESULTS\n", ("header", "center"))
        self.rf_output_text.insert(tk.END, f"{team1} vs {team2}\n\n", ("teams", "center"))

        
        # Winner with confidence bar
        confidence_bar = "▓" * int(confidence * 10) + "|" * (10 - int(confidence * 10))
        self.rf_output_text.insert(tk.END, f"Predicted Winner: ", "default")
        self.rf_output_text.insert(tk.END, f"{winner} ", "winner")
        self.rf_output_text.insert(tk.END, f"{confidence:.1%}\n", "confidence")
        self.rf_output_text.insert(tk.END, f"{confidence_bar}\n\n", "confidence_bar")
        
        # Stats table
        stats = [
            ('Win %', 'W_PCT', '0.3f'),
            ('Points/G', 'PTS', '0.1f'),
            ('Rebounds/G', 'REB', '0.1f'),
            ('Assists/G', 'AST', '0.1f'),
            ('FG %', 'FG_PCT', '0.3f'),
            ('3P %', 'FG3_PCT', '0.3f')
        ]
        
        self.rf_output_text.insert(tk.END, "KEY STATS COMPARISON\n", "stats_header")
        self.rf_output_text.insert(tk.END, f"{'Stat':<14}{team1[:12]:<12}{team2[:12]:<12}\n", "column_header")
        
        for stat, col, fmt in stats:
            self.rf_output_text.insert(tk.END, 
                f"{stat:<14}{getattr(t1_stats, col):{fmt}}{getattr(t2_stats, col):>12.{fmt[2:]}}\n",
                "stats")
        
        self.rf_output_text.config(state="disabled")

    def configure_text_tags(self):
        """Configure all text styling tags"""
        # Justification
        self.rf_output_text.tag_config("center", justify="center")
        
        # Other tags
        self.rf_output_text.tag_config("header", font=('Helvetica', 18, 'bold'))
        self.rf_output_text.tag_config("teams", font=('Helvetica', 16))
        self.rf_output_text.tag_config("winner", font=('Helvetica', 16, 'bold'))
        self.rf_output_text.tag_config("confidence")
        self.rf_output_text.tag_config("confidence_bar")
        self.rf_output_text.tag_config("stats_header", font=('Helvetica', 14, 'bold'))
        self.rf_output_text.tag_config("column_header", font=('Helvetica', 14, 'bold'))
        self.rf_output_text.tag_config("stats", font=('Courier New', 15))
    
    
    def create_navbar(self):
        """Creates the top navigation bar"""
        navbar = tb.Frame(self.main_frame, bootstyle=PRIMARY)
        navbar.pack(fill=tk.X, pady=0)

        # Navigation buttons
        btn_teams = tb.Button(navbar, text="NBA TEAMS", bootstyle="success",
                                 command=self.show_teams_window)
        btn_teams.pack(side=tk.LEFT, padx=15, pady=10)
        
        btn_standing = tb.Button(navbar, text="NBA PLAYER STANDINGS", bootstyle="success", 
                                command=self.show_player_standings_window)
        btn_standing.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Central title
        lbl_title = tb.Label(navbar, text="NBA GAME PREDICTOR", 
                             font=("Arial", 14), bootstyle="light-secondary")
        lbl_title.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Stats button
        btn_stats = tb.Button(navbar, text="GENERAL STATS", bootstyle="success")
        btn_stats.pack(side=tk.RIGHT, padx=15, pady=10)

    def show_teams_window(self):
        """Create and show the teams window"""
        TeamsWindow(self, self.nba_teams)

    def show_player_standings_window(self):
        SimplePlayerStandings(self)

    def create_main_content(self):
        """Creates the main content of the application"""
        content_frame = tb.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame (teams list)
        left_frame = tb.Frame(content_frame, bootstyle="secondary")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # List title
        lbl_teams = tb.Label(left_frame, text="NBA Teams List", 
                             font=("Arial", 12), bootstyle="secondary")
        lbl_teams.pack(pady=10)
        
        # Teams list with scrollbar
        teams_frame = tb.Frame(left_frame)
        teams_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        teams_scrollbar = tb.Scrollbar(teams_frame)
        teams_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        teams_listbox = tk.Listbox(teams_frame, width=25, height=30, 
                                        yscrollcommand=teams_scrollbar.set)
        teams_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        teams_scrollbar.config(command=teams_listbox.yview)
        
        # Fill teams list
        for team in self.nba_teams:
            teams_listbox.insert(tk.END, team)
        
        # Center frame (predictions) - Modified to only contain the prediction results
        center_frame = tb.Frame(content_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

                # Team selection frame
        select_frame = tb.Frame(center_frame)
        select_frame.pack(pady=10)

        # Team 1 dropdown
        tb.Label(select_frame, text="Team 1:", font=("Arial", 10)).grid(row=0, column=0, padx=5, pady=5)
        self.team1_var = tk.StringVar()
        team1_menu = tb.Combobox(select_frame, textvariable=self.team1_var, values=self.nba_teams, width=25)
        team1_menu.grid(row=0, column=1, padx=5, pady=5)

        # Team 2 dropdown
        tb.Label(select_frame, text="Team 2:", font=("Arial", 10)).grid(row=1, column=0, padx=5, pady=5)
        self.team2_var = tk.StringVar()
        team2_menu = tb.Combobox(select_frame, textvariable=self.team2_var, values=self.nba_teams, width=25)
        team2_menu.grid(row=1, column=1, padx=5, pady=5)

        # Predict button
        predict_btn = tb.Button(select_frame, text="Predict Winner", bootstyle="info", command=self.make_prediction)
        predict_btn.grid(row=2, column=0, columnspan=2, pady=10)

        
        # Prediction section
        prediction_label = tb.Label(center_frame, text="GAME PREDICTION RESULTS", 
                                         font=("Arial", 12, "bold"), bootstyle="primary")
        prediction_label.pack(pady=20)
        
        # Prediction area with Scrollbar
        prediction_frame = tb.Frame(center_frame, bootstyle="default")
        prediction_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        prediction_frame.config(relief="groove", borderwidth=2)
        
        # Add scrollbar
        scrollbar = tb.Scrollbar(prediction_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget for Random Forest output
        self.rf_output_text = tk.Text(prediction_frame, wrap=tk.NONE,
                                    yscrollcommand=scrollbar.set,
                                    font=("Courier", 9))  # Use monospace font for alignment
        self.rf_output_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.rf_output_text.yview)

        # Right frame (past results)
        right_frame = tb.Frame(content_frame, bootstyle="secondary")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Results title
        lbl_results = tb.Label(right_frame, text="Past Game Results", 
                                 font=("Arial", 12), bootstyle="secondary", anchor="center")
        lbl_results.pack(pady=10)
        
        # Results list with scrollbar
        results_frame = tb.Frame(right_frame)
        results_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        results_scrollbar = tb.Scrollbar(results_frame)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Replace Listbox with Text widget for better formatting
        self.results_text = tk.Text(results_frame, width=30, height=30,
                                        yscrollcommand=results_scrollbar.set, wrap=tk.WORD,
                                        font=("Arial", 9))
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.config(command=self.results_text.yview)
        
        self.results_text.insert(tk.END, "Loading past game results...\n")
        self.results_text.config(state="disabled")

    
    def load_playoff_data(self):
        """Load NBA playoff data in a separate thread to avoid UI freezing"""
        self.playoff_data = self.fetch_nba_playoff_gamelog()
        if self.playoff_data:
            self.processed_data = self.process_game_log_data(self.playoff_data)
            
            # Use after() to safely update UI from the main thread
            self.after(100, self.update_results_display)
    
    def update_results_display(self):
        """Update the results display with the playoff data"""
        if self.processed_data is not None:
            self.display_game_log_in_ui()
        else:
            self.results_text.config(state="normal")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Failed to load NBA playoff game data.")
            self.results_text.config(state="disabled")


    def fetch_nba_playoff_gamelog(self):
        """Fetch NBA playoff game log data from the NBA API"""
        url = "https://stats.nba.com/stats/leaguegamelog"

        params = {
            "Counter": "1000",
            "DateFrom": "",
            "DateTo": "",
            "Direction": "DESC",
            "ISTRound": "",
            "LeagueID": "00",
            "PlayerOrTeam": "T",
            "Season": "2024-25",
            "SeasonType": "Playoffs",
            "Sorter": "DATE"
        }

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nba.com/",
            "Accept-Language": "en-US,en;q=0.9"
        }

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.after(100, lambda: self.update_error_message(f"Error fetching NBA data: {e}"))
            return None

    def update_error_message(self, message):
        """Update the results text with an error message"""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, message)
        self.results_text.config(state="disabled")

    def format_date(self, date_str):
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
            return date_obj.strftime("%b %d, %Y")
        except:
            return date_str

    def process_game_log_data(self, data):
        if not data or "resultSets" not in data:
            return None

        result_sets = data["resultSets"]
        if not result_sets or len(result_sets) == 0:
            return None

        game_log = result_sets[0]
        headers = game_log.get("headers", [])
        rows = game_log.get("rowSet", [])

        df = pd.DataFrame(rows, columns=headers)
        return df

    def display_game_log_in_ui(self):
        """Display game log in the UI text widget"""
        df = self.processed_data
        
        if df is None or df.empty:
            self.results_text.config(state="normal")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No game log data available")
            self.results_text.config(state="disabled")
            return

        display_cols = [
            "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "TEAM_NAME", "PTS", 
            "FG_PCT", "FG3_PCT", "REB", "AST", "STL", "BLK", "TOV"
        ]

        existing_cols = [col for col in display_cols if col in df.columns]
        if not existing_cols:
            self.results_text.config(state="normal")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "None of the expected columns found in the data")
            self.results_text.config(state="disabled")
            return

        display_df = df[existing_cols].copy()

        if "GAME_DATE" in display_df.columns:
            display_df["GAME_DATE"] = display_df["GAME_DATE"].apply(self.format_date)

        if "GAME_ID" in display_df.columns:
            unique_games = display_df["GAME_ID"].unique()
            last_10_games = unique_games[:10]
            display_df = display_df[display_df["GAME_ID"].isin(last_10_games)]

        # Update UI
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        # Title with tag for styling
        self.results_text.tag_configure("title", font=("Arial", 10, "bold"))
        self.results_text.tag_configure("subtitle", font=("Arial", 9, "bold"))
        self.results_text.tag_configure("game_number", font=("Arial", 9, "bold"), foreground="#3498db")
        self.results_text.tag_configure("win", foreground="#2ecc71")
        self.results_text.tag_configure("loss", foreground="#e74c3c")
        
        self.results_text.insert(tk.END, "LAST 10 NBA PLAYOFF GAMES\n", "title")
        self.results_text.insert(tk.END, "2024-25 SEASON\n", "subtitle")
        self.results_text.insert(tk.END, "----------------------------\n")

        for i, (_, row) in enumerate(display_df.iterrows()):
            self.results_text.insert(tk.END, f"\nGame {i+1}", "game_number")
            self.results_text.insert(tk.END, f" - {row.get('GAME_DATE', 'Date unknown')}\n")
            
            # Apply win/loss styling
            wl_tag = "win" if row.get('WL') == "W" else "loss"
            
            self.results_text.insert(tk.END, f"{row.get('TEAM_NAME', 'Team')} ")
            self.results_text.insert(tk.END, f"{row.get('WL', '')}", wl_tag)
            self.results_text.insert(tk.END, f": {row.get('PTS', 0)} pts\n")
            self.results_text.insert(tk.END, f"{row.get('MATCHUP', '')}\n")
            
            stats = (
                f"FG%: {row.get('FG_PCT', 0):.3f} | "
                f"3P%: {row.get('FG3_PCT', 0):.3f}\n"
                f"REB: {row.get('REB', 0)} | "
                f"AST: {row.get('AST', 0)} | "
                f"STL: {row.get('STL', 0)} | "
                f"BLK: {row.get('BLK', 0)} | "
                f"TOV: {row.get('TOV', 0)}\n"
            )
            self.results_text.insert(tk.END, stats)
            
        # Add series tracker
        self.display_series_tracker_in_ui(df)
        
        self.results_text.config(state="disabled")

    def display_series_tracker_in_ui(self, df):
        """Display playoff series tracker in the UI text widget"""
        if df is None or df.empty or "MATCHUP" not in df.columns or "WL" not in df.columns:
            return

        self.results_text.insert(tk.END, "\n\nPLAYOFF SERIES TRACKER\n", "title")
        self.results_text.insert(tk.END, "----------------------\n")

        if "GAME_ID" in df.columns:
            unique_games = df["GAME_ID"].unique()
            last_10_games = unique_games[:10]
            df = df[df["GAME_ID"].isin(last_10_games)]

        series_dict = {}
        
        recent_game_ids = df["GAME_ID"].drop_duplicates().head(10)
        df = df[df["GAME_ID"].isin(recent_game_ids)]


        for _, row in df.iterrows():
            matchup = row["MATCHUP"]
            team = row["TEAM_NAME"]
            wl = row["WL"]

            opponent = None
            if "@" in matchup:
                teams = matchup.split(" @ ")
                if len(teams) == 2:
                    opponent = teams[1]
            elif "vs." in matchup:
                teams = matchup.split(" vs. ")
                if len(teams) == 2:
                    opponent = teams[1]

            if team and opponent:
                series_teams = sorted([team, opponent])
                series_key = f"{series_teams[0]} vs {series_teams[1]}"

                if series_key not in series_dict:
                    series_dict[series_key] = {}

                if team not in series_dict[series_key]:
                    series_dict[series_key][team] = 0
                if opponent not in series_dict[series_key]:
                    series_dict[series_key][opponent] = 0

                if wl == "W":
                    series_dict[series_key][team] += 1

        for series, scores in series_dict.items():
            teams = list(scores.keys())
            if len(teams) == 2:
                self.results_text.insert(tk.END, f"\n{series}:\n")
                self.results_text.insert(tk.END, f"{teams[0]} {scores[teams[0]]} - {scores[teams[1]]} {teams[1]}\n")

class TeamsWindow(tb.Toplevel):
    def __init__(self, parent, teams):
        super().__init__(parent)
        self.title("NBA Teams List")
        self.geometry("400x600")
        
        # Create main frame
        main_frame = tb.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        lbl_title = tb.Label(main_frame, text="NBA Teams", 
                            font=("Arial", 14, "bold"), bootstyle="primary")
        lbl_title.pack(pady=10)
        
        # Teams list with scrollbar
        list_frame = tb.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tb.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.teams_listbox = tk.Listbox(list_frame, width=30, height=25,
                                            yscrollcommand=scrollbar.set,
                                            font=("Arial", 12))
        self.teams_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.teams_listbox.yview)
        
        # Fill teams list
        for team in teams:
            self.teams_listbox.insert(tk.END, team)
            
class SimplePlayerStandings(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("NBA Player Standings")
        self.geometry("900x600")

        self.season_var = tk.StringVar(value="2024-25")
        self.stat_var = tk.StringVar(value="PTS")

        self.create_widgets()
        self.fetch_data()


    def create_widgets(self):
        # Controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Season:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(control_frame, textvariable=self.season_var,
                     values=["2024-25", "2023-24", "2022-23"], width=10).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="Stat:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(control_frame, textvariable=self.stat_var,
                     values=["PTS", "REB", "AST", "STL", "BLK"], width=10).pack(side=tk.LEFT)

        ttk.Button(control_frame, text="Refresh", command=self.fetch_data).pack(side=tk.LEFT, padx=10)

        # Treeview
        self.tree = ttk.Treeview(self, show='headings')
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def fetch_data(self):
        url = "https://stats.nba.com/stats/leagueLeaders"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nba.com/",
            "x-nba-stats-origin": "stats",
            "x-nba-stats-token": "true"
        }
        params = {
            "LeagueID": "00",
            "PerMode": "PerGame",
            "Scope": "S",
            "Season": self.season_var.get(),
            "SeasonType": "Regular Season",
            "StatCategory": self.stat_var.get()
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            headers_ = data["resultSet"]["headers"]
            rows = data["resultSet"]["rowSet"]

            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = headers_

            for col in headers_:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=80)

            for row in rows:
                self.tree.insert("", tk.END, values=row)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch data:\n{e}")

class TeamsWindow2(tb.Toplevel):
    def __init__(self, parent, home_team, away_team):
        super().__init__(parent)
        self.title(f"Stats: {home_team} vs {away_team}")
        self.geometry("1500x150")

        

        self.main_frame = tb.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.lbl_title = tb.Label(self.main_frame, text=f"{home_team} vs {away_team}",
                                  font=("Arial", 14, "bold"), bootstyle="primary")
        self.lbl_title.pack(pady=10)

        self.list_frame = tb.Frame(self.main_frame)
        self.list_frame.pack(fill=tk.BOTH, expand=True)

        # Ejecutar query y obtener DataFrame
        df = self.url(home_team, away_team)

        if df is not None:
            self.columns = list(df.columns)
            self.teams_tree = ttk.Treeview(self.list_frame, columns=self.columns, show='headings')

            for col in self.columns:
                self.teams_tree.heading(col, text=col)
                self.teams_tree.column(col, anchor="center", stretch=False, width=50)


            for _, row in df.iterrows():
                self.teams_tree.insert("", tk.END, values=list(row))

            self.scrollbar_y = ttk.Scrollbar(self.list_frame, orient=tk.VERTICAL, command=self.teams_tree.yview)
            self.scrollbar_x = ttk.Scrollbar(self.list_frame, orient=tk.HORIZONTAL, command=self.teams_tree.xview)
            self.teams_tree.configure(yscroll=self.scrollbar_y.set, xscroll=self.scrollbar_x.set)

            self.teams_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
            self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        else:
            tb.Label(self.list_frame, text="No se encontraron datos.",
                     bootstyle="danger").pack()
        
if __name__ == "__main__":
    app = NBAPredictor()
    app.mainloop()