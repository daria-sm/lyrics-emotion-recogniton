import pandas as pd

# Assuming you have your original DataFrame
# original_df = ...
original_df = pd.read_json("../data/processed/limit-all.json", lines=True)
# Extract the first 10 records

# Load your original dataset into a Pandas DataFrame
# original_df = pd.read_csv('your_dataset.csv')

# Generate a random sample representing 70% of the data
sample_percentage = 0.7
random_sample = original_df.sample(frac=sample_percentage, random_state=42)

# Use the drop method to get the remaining 30% of data
remaining_data = original_df.drop(random_sample.index)

# Now, 'random_sample' contains 70% of random rows, and 'remaining_data' contains the remaining 30%.




with open("../data/processed/limit-70-per-cent.json", 'w') as file:
    for _, row in random_sample.iterrows():
        json_row = row.to_json()
        file.write(json_row + '\n')

with open("../data/processed/limit-30-per-cent.json", 'w') as file:
    for _, row in remaining_data.iterrows():
        json_row = row.to_json()
        file.write(json_row + '\n')