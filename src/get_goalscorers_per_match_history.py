import pandas as pd
import sqlite3

# Load the merged CSV file
df = pd.read_csv("results_with_goals.csv")

# Create an SQLite connection (in-memory or a database file)
conn = sqlite3.connect(":memory:")  # Use ":memory:" for temporary storage, or "database.db" for persistence.

# Store the DataFrame as a table in SQLite
df.to_sql("merged_table", conn, if_exists="replace", index=False)

# Run an SQL query (Example: Selecting specific columns where a condition is met)
home_team = input('Enter home team name: ')
away_team = input('Enter away team name: ')
query1 = "SELECT scorer, team, COUNT(minute) as goals FROM merged_table WHERE home_team = ? and away_team = ? and tournament = 'FIFA World Cup qualification' GROUP BY scorer, team ORDER BY goals DESC"
result_df = pd.read_sql(query1, conn, params=(home_team, away_team))

query2 = "SELECT * FROM merged_table WHERE home_team = ? and away_team = ? and tournament = 'FIFA World Cup qualification'"
result_df_2 = pd.read_sql(query2, conn, params=(home_team, away_team))

# Display results
print(result_df)
print(result_df_2)

# Close the connection
conn.close()
