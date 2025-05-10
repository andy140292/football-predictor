import pandas as pd

# Load the CSV files
results = pd.read_csv("results.csv")
goalscorers = pd.read_csv("goalscorers.csv")

# Define the common columns to join on
common_columns = ["date", "home_team", "away_team"]

# Perform the merge (inner join by default)
results_with_goals = pd.merge(results, goalscorers, on=common_columns, how="inner")

# Save the merged dataframe to a new CSV file
results_with_goals.to_csv("results_with_goals.csv", index=False)

print("Merged file saved as 'merged_output.csv'")
