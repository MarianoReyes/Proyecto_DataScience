import pandas as pd

# Load the data from the merged_data.csv file
df = pd.read_csv('data/merged_data.csv')

# Define a function to normalize a column
def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)

# Normalize the "content" and "wording" columns
df['normalized_content'] = normalize_column(df['content'])
df['normalized_wording'] = normalize_column(df['wording'])

# Save the normalized data to a new CSV file
df.to_csv('data/normalized_merged_data.csv', index=False)

print("Normalized data saved to normalized_merged_data.csv")
