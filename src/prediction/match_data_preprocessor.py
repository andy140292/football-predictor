import pandas as pd

class MatchDataPreprocessor:
    def __init__(self, file_path, south_america_teams):
        """
        Initializes the preprocessor with the dataset file path and list of South American teams.
        """
        self.file_path = file_path
        self.south_america_teams = south_america_teams
        self.matches = None  # Placeholder for the dataset
    
    def load_and_filter_data(self):
        """Loads the dataset and filters for FIFA World Cup qualification matches in South America."""
        self.matches = pd.read_csv(self.file_path)

        # Filter for specific tournaments
        self.matches = self.matches[self.matches['tournament'].isin([
                'FIFA World Cup qualification',
                'Copa America',
                'FIFA World Cup',
                'Friendly'
            ])
        ]
        
        # Filter matches involving South American teams
        self.matches = self.matches[self.matches['home_team'].isin(self.south_america_teams)]
        self.matches = self.matches[self.matches['away_team'].isin(self.south_america_teams)]

        # Convert 'date' column to datetime format
        self.matches['date'] = pd.to_datetime(self.matches['date'])

        # Filter matches played after the year 2000
        self.matches = self.matches[self.matches['date'] >= '2000-01-01']
    
    def create_matchup_identifier(self):
        """Creates a unique identifier for each matchup (regardless of home/away status)."""
        self.matches['matchup_id'] = self.matches.apply(
            lambda row: '_'.join(sorted([row['home_team'], row['away_team']])), axis=1
        )

    def merge_head_to_head_stats(self, head_to_head_stats):
        """Merges head-to-head statistics into the dataset."""
        self.matches = pd.merge(self.matches, head_to_head_stats, on='matchup_id', how='left')

        # Fill missing values (for matchups with no history)
        self.matches['head_to_head_goal_diff'] = self.matches['head_to_head_goal_diff'].fillna(0)

    def calculate_rolling_averages(self, window_size=10):
        """Computes rolling averages for goals scored and conceded for home and away teams."""
        self.matches['home_team_avg_scored'] = self.matches.groupby('home_team')['home_score'] \
            .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
        self.matches['home_team_avg_conceded'] = self.matches.groupby('home_team')['away_score'] \
            .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())

        self.matches['away_team_avg_scored'] = self.matches.groupby('away_team')['away_score'] \
            .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
        self.matches['away_team_avg_conceded'] = self.matches.groupby('away_team')['home_score'] \
            .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())

    def encode_teams(self):
        """Converts categorical team names into dummy variables (one-hot encoding)."""
        self.matches = pd.get_dummies(self.matches, columns=['home_team', 'away_team'])

    def encode_confederations(self):
        """Converts categorical team names into dummy variables (one-hot encoding)."""
        self.matches = pd.get_dummies(self.matches, columns=['home_team_confederation', 'away_team_confederation'])        

    # Define the new multi-class target variable
    def get_match_outcome(self, row):
        if row['home_score'] > row['away_score']:
            return 2  # Home Win
        elif row['home_score'] < row['away_score']:
            return 0  # Away Win
        else:
            return 1  # Draw

    def finalize_dataset(self):
        """Cleans NaN values, resets indices, and prepares X (features) and y (target)."""
        self.matches = self.matches.sort_values('date')  # Sort by date

        # Drop rows with missing values
        self.matches = self.matches.dropna().reset_index(drop=True)

        # Define feature matrix X and target variable y
        X = self.matches.drop(columns=['date', 'home_score', 'away_score', 'tournament', 'neutral', 'city', 'country', 'matchup_id', 'goal_diff'])
        y = self.matches.apply(self.get_match_outcome, axis=1)  # Apply match outcome function

        return X, y
    
    def get_head_to_head_stats(self):
 
        # Calculate goal difference for each match (from the perspective of the listed home team)
        self.matches['goal_diff'] = self.matches['home_score'] - self.matches['away_score']

        # Group by matchup_id and compute the average goal difference
        head_to_head_stats = self.matches.groupby('matchup_id')['goal_diff'].mean().reset_index()
        head_to_head_stats.rename(
            columns={'goal_diff': 'head_to_head_goal_diff'}, inplace=True
        )
        return head_to_head_stats

    def preprocess(self):
        """Runs the entire preprocessing pipeline and returns cleaned X, y datasets."""
        self.load_and_filter_data()
        self.create_matchup_identifier()
        head_to_head_stats = self.get_head_to_head_stats()
        self.merge_head_to_head_stats(head_to_head_stats)
        self.calculate_rolling_averages()
        self.encode_teams()
        self.encode_confederations()
        return self.finalize_dataset()
