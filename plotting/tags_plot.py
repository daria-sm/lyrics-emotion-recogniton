import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
df = pd.read_csv('../data/processed/tags_vad_q_sorted.csv')

# Filter the DataFrame to exclude data points where Valence or Arousal is equal to 0.5
filtered_df = df[(df['Valence'] != 0.5) & (df['Arousal'] != 0.5)]

# Create the scatter plot using the filtered data
plt.scatter(x=filtered_df['Valence'], y=filtered_df['Arousal'])

plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1.0])
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1.0])

# Set the origin of the axes at (0.5, 0.5)
plt.axhline(y=0.5, color='k', linestyle='--', linewidth=0.5, )
plt.axvline(x=0.5, color='k', linestyle='--', linewidth=0.5)

# Add labels for Valence and Arousal outside the graph with better positioning
plt.text(0.48, 1.05, 'A', fontsize=12, color='black', transform=plt.gca().transAxes)
plt.text(1.05, 0.48, 'V', fontsize=12, color='black', transform=plt.gca().transAxes)

# Save the figure as an image file (e.g., PNG)
plt.savefig('plot_tags.png')

# Show the plot
plt.show()
