import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Set seaborn style for a cleaner, modern look
sns.set_theme(style="whitegrid", palette="muted")

# Create a figure with multiple subplots for different graphs
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# --- Plot 1: Scatter plot of Accuracy vs. Temperature ---
# This plot distinguishes between 'Checklist' types and 'determina_in' values.
sns.scatterplot(
    data=df,
    x='Temperature',
    y='accuracy',
    hue='Checklist',
    style='determina_in',
    s=100,
    ax=axes[0, 0]
)
axes[0, 0].set_title('Accuracy vs Temperature')
axes[0, 0].set_xlabel('Temperature')
axes[0, 0].set_ylabel('Accuracy')

# --- Plot 2: Boxplot of F1 Score by Checklist ---
sns.boxplot(
    data=df,
    x='Checklist',
    y='f1_score',
    ax=axes[0, 1]
)
axes[0, 1].set_title('F1 Score Distribution by Checklist')
axes[0, 1].set_xlabel('Checklist')
axes[0, 1].set_ylabel('F1 Score')

# --- Plot 3: Line plot of Balanced Accuracy vs. Temperature grouped by Modello ---
sns.lineplot(
    data=df,
    x='Temperature',
    y='balanced_accuracy',
    hue='Modello',
    marker='o',
    ax=axes[1, 0]
)
axes[1, 0].set_title('Balanced Accuracy vs Temperature by Modello')
axes[1, 0].set_xlabel('Temperature')
axes[1, 0].set_ylabel('Balanced Accuracy')

# --- Plot 4: Bar plot of Percentage Equal per Determina (rotated for readability) ---
sns.barplot(
    data=df,
    x='Determina',
    y='percentage_equal',
    hue='Checklist',
    ax=axes[1, 1]
)
axes[1, 1].set_title('Percentage Equal per Determina')
axes[1, 1].set_xlabel('Determina')
axes[1, 1].set_ylabel('Percentage Equal')
axes[1, 1].tick_params(axis='x', rotation=45)

# Adjust layout for better spacing and display the plots
plt.tight_layout()
plt.show()