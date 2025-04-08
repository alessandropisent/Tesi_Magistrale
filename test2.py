import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# 1. Reconstruct the data into a pandas DataFrame (same data as before)
data = {
    'Municipality': ['Lucca'] * 24,
    'Temperature': [
        0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.20, 0.20, 0.20,
        0.40, 0.40, 0.40, 0.50, 0.50, 0.50, 0.60, 0.60, 0.60,
        0.80, 0.80, 0.80, 1.00, 1.00, 1.00
    ],
    'Model': [
        'chatgpt 4o mini', 'llama 3.1 8B', 'llama 3.2 3B',
        'chatgpt 4o mini', 'llama 3.1 8B', 'llama 3.2 3B',
        'chatgpt 4o mini', 'llama 3.1 8B', 'llama 3.2 3B',
        'chatgpt 4o mini', 'llama 3.1 8B', 'llama 3.2 3B',
        'chatgpt 4o mini', 'llama 3.1 8B', 'llama 3.2 3B',
        'chatgpt 4o mini', 'llama 3.1 8B', 'llama 3.2 3B',
        'chatgpt 4o mini', 'llama 3.1 8B', 'llama 3.2 3B',
        'chatgpt 4o mini', 'llama 3.1 8B', 'llama 3.2 3B'
    ],
    'accuracy': [
        1.000000, 0.666667, 0.666667, 1.000000, 0.666667, 0.666667,
        1.000000, 0.666667, 0.666667, 0.666667, 0.666667, 0.666667,
        0.666667, 0.666667, 0.666667, 0.666667, 0.666667, 0.666667,
        0.666667, 1.000000, 0.666667, 0.666667, 0.666667, 0.666667
    ]
}
df = pd.DataFrame(data)

# 2. Create the grouped bar plot using Seaborn
plt.figure(figsize=(15, 8)) # Use a wider figure for grouped bars
sns.set_theme(style="whitegrid") # Apply a theme

# Create the bar plot
# x='Temperature' defines the positions of the groups
# y='accuracy' defines the height of the bars
# hue='Model' defines the individual bars within each group by color
barplot = sns.barplot(
    data=df,
    x='Temperature',
    y='accuracy',
    hue='Model',
    palette='viridis' # Example using the 'viridis' color palette
    # ci=None # Use errorbar=None in newer seaborn versions if you don't want error bars
)

# 3. Customize the plot
municipality_name = df['Municipality'].iloc[0]
plt.title(f'Model Accuracy per Temperature Setting in {municipality_name}', fontsize=16, pad=20)
plt.xlabel('Temperature Setting', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# Adjust y-axis limits to better show variation, but starting near 0 is often good for bars
# Let's start slightly below the minimum accuracy observed (around 0.66) to give some space
min_accuracy = df['accuracy'].min()
plt.ylim(min_accuracy - 0.1, 1.05) # Start just below minimum, end just above maximum

# Optional: Add value labels on top of bars
# This can get crowded on grouped bar charts, adjust fontsize/fmt as needed
# for container in barplot.containers:
#    barplot.bar_label(container, fmt='%.2f', fontsize=9, padding=3)


# Adjust legend position
plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

# 4. Show the plot
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout rect to make space for legend
plt.show()