import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from tabulate import tabulate
import os

plot_dir = "Graphs"
data = pd.read_csv("result_choose_accuracy.csv")

#data_p = data[["accuracy","Model","Temperature"]].groupby(["Model","Temperature"]).mean()

data_p = data.pivot_table(index='Model',
                          columns='Temperature',
                          values='accuracy')
# --- Optional: Format the output for better readability ---
# Sort columns (Temperatures) numerically
pivot_df = data_p.sort_index(axis=1)

# Round the accuracy values to, for example, 4 decimal places
#pivot_df_formatted = pivot_df.round(4)


print("\n" + "="*40 + "\n")
print("LaTeX formatted table:")
# makes 'Model' a column again for tabulate
with open("table.txt","w") as f:
    f.write(tabulate(pivot_df.reset_index(), headers='keys', tablefmt='latex_booktabs', showindex=False, floatfmt=".2f"))

print(pivot_df)

df = pd.read_csv("src/Evaluation/checklist_compiler/statistics.csv")

metrics = ["accuracy","precision","recall","f1_score","balanced_accuracy"]

for metric in metrics:
    data_p = df.pivot_table(index='Modello',
                          columns='Temperature',
                          values=metric)
    pivot_df = data_p.sort_index(axis=1)
    with open(f"answer_table_{metric}.txt","w") as f:
        f.write(tabulate(pivot_df.reset_index(), headers='keys', tablefmt='latex_booktabs', showindex=False, floatfmt=".2f"))

temp0 = df[df["Temperature"]==0.0]
temp0 = temp0[["Modello","accuracy","precision","recall","f1_score","balanced_accuracy"]].groupby("Modello").mean().reset_index().sort_values(by="accuracy",ascending=False)

print("="*50)
print("\n"*3)
temp0 = temp0.reset_index().drop(columns=["index"])
print(temp0)

with open(f"answer_table_recap.txt","w") as f:
    f.write(tabulate(temp0, headers='keys', tablefmt='latex_booktabs', showindex=False, floatfmt=".2f"))

# Helper function to create a custom palette based on model families
def create_family_palette(model_list):
    """
    Generates a seaborn color palette dictionary mapping model names to colors,
    grouping by 'llama', 'mistral', and 'gpt' families.
    """
    family_colors = {
        'llama': 'royalblue',    # Base color for Llama family
        'mistral': 'darkorange', # Base color for Mistral family
        'gpt': 'forestgreen',    # Base color for GPT family
    }
    other_color = 'grey'         # Color for models not in these families

    # Categorize models by family
    models_by_family = {'llama': [], 'mistral': [], 'gpt': [], 'other': []}
    # Sort the model list to ensure consistent color assignment across plots
    # Use a stable sort like mergesort (pandas default) or Timsort (Python default)
    # or just Python's sorted() which is stable.
    sorted_model_list = sorted(list(model_list))

    for model in sorted_model_list:
        model_lower = model.lower()
        if 'llama' in model_lower:
            models_by_family['llama'].append(model)
        elif 'mistral' in model_lower:
            models_by_family['mistral'].append(model)
        elif 'gpt' in model_lower:
            models_by_family['gpt'].append(model)
        else:
            # Handle potential future models or unexpected names
            models_by_family['other'].append(model)

    final_palette = {}

    # Generate specific shades for each model within its family
    for family, models in models_by_family.items():
        if not models: # Skip if no models in this family
            continue

        n_colors = len(models)
        base_color = family_colors.get(family, other_color)

        # Generate a palette of shades for the family
        # Using light_palette to get variations. Add n_colors+1 and slice [1:]
        # to avoid the lightest color if it's too pale, especially for single models.
        # Reverse=True makes the 'first' model (e.g., smallest size) darker usually.
        if n_colors == 1:
            # If only one model, just use the base color
             family_palette_colors = [base_color]
        elif family == 'other':
             # Use sequential grey palette if multiple 'other' models
             family_palette_colors = sns.color_palette("Greys", n_colors=n_colors + 2)[1:-1] # Avoid pure white/black
        else:
            # Use light_palette for main families to get shades
            # Adjust reverse=True/False based on desired shade order
            family_palette_colors = sns.light_palette(base_color, n_colors=n_colors + 1, reverse=False)[1:]

        # Assign the generated colors to the specific models in the sorted list
        for model, color in zip(models, family_palette_colors):
            final_palette[model] = color

    return final_palette

# Melt the DataFrame to long format, suitable for seaborn plotting
df_melted = temp0.melt(id_vars='Modello', var_name='Metric', value_name='Value')

# Sort models for consistent letter assignment and plotting order
# This MUST match the sorting used in create_family_palette
sorted_models = sorted(temp0['Modello'].unique())

# Create mapping from model name to letter
model_to_letter = {model: letter for letter, model in zip(string.ascii_uppercase, sorted_models)}
letter_to_model = {v: k for k, v in model_to_letter.items()} # For legend

# Create the custom color palette using the helper function
model_palette = create_family_palette(sorted_models) # Use sorted_models here too

# --- Plotting ---
# Set plot style and context for smaller fonts suitable for papers/LaTeX
sns.set_theme(style="whitegrid", context="paper") # 'paper' context uses smaller fonts

# Create the catplot (facet grid of bar plots)
# Adjust height, aspect, and col_wrap for LaTeX page width (approx 6.5 inches)
g = sns.catplot(
    data=df_melted,
    kind="bar",
    x="Modello",           # Models on x-axis (will be replaced by letters)
    y="Value",             # Score on y-axis
    col="Metric",          # Facets based on Metric
    hue="Modello",         # Color bars by model name (needed for palette and legend handles)
    hue_order=sorted_models, # *** Explicitly set hue order for reliable handle mapping ***
    order=sorted_models,   # *** Explicitly set x-axis order to match hue ***
    palette=model_palette, # Use the custom generated palette
    height=2.8,            # Height of each facet (inches)
    aspect=0.8,            # Aspect ratio
    col_wrap=3,            # Wrap facets into rows
    legend=False,          # Turn off automatic seaborn legend
    dodge=False            # Ensure bars align with x-ticks directly (since hue=x)
)

# --- Customization for LaTeX ---

# Add letter labels to bars (at the bottom)
for ax in g.axes.flat:
    # Check if axes has patches (bars) before proceeding
    if not ax.patches:
        continue

    # Iterate through bars (patches) and add text
    # The order of patches should now match sorted_models due to `order` and `hue_order`
    for i, patch in enumerate(ax.patches):
        if i < len(sorted_models):
            model_name = sorted_models[i] # Get model name based on index
            letter = model_to_letter.get(model_name, '?') # Get letter

            # Calculate position for the text label (at the bottom of the bar)
            x = patch.get_x() + patch.get_width() / 2
            y_pos = 0.01 # Small offset from the bottom baseline (y=0)
            ax.text(x, y_pos, letter,
                    ha='center', va='bottom', # Center horizontally, align bottom of text to y_pos
                    fontsize=7, color='black', fontweight='bold') # Make letter visible
        else:
            # This case should be less likely now with explicit ordering
            print(f"Warning: Patch index {i} out of bounds for sorted_models on axis '{ax.get_title()}'.")


# Set x-axis labels to empty strings as letters are now on bars
g.set_xticklabels([])
g.set(xlabel=None) # Remove "Modello" label from x-axis

# Set y-axis limits
g.set(ylim=(0, 1.05)) # Keep ylim starting at 0

# Add titles and labels with adjusted font sizes
g.figure.suptitle('Model Performance Comparison Across Metrics (Temp 0.0)', y=1.04, fontsize=12) # Adjust vertical position and size
g.set_axis_labels("", "Score", fontsize=10) # Remove x-axis label, keep y-axis label
g.set_titles("Metric: {col_name}", fontsize=10) # Titles for each facet
g.tick_params(axis='y', which='major', labelsize=8) # Adjust y-tick label size
g.tick_params(axis='x', bottom=False) # Remove x-axis ticks

# --- Add Custom Legend ---
# Get handles (colors/patches) from one of the axes.
# Thanks to hue_order, handles should correspond to sorted_models.
handles = g.axes[0].patches[:len(sorted_models)] # Get the bar patches directly

# Create new labels mapping Letter: Model Name
legend_labels = [f"{model_to_letter[model]}: {model}" for model in sorted_models]

# Place legend below the center, adjust anchor, number of columns, and font sizes
# Increase the negative offset slightly if needed, ensure bottom margin is large enough
g.figure.legend(handles, legend_labels,
                title='Model Key',
                loc='lower center', # Center horizontally below axes
                bbox_to_anchor=(0.5, -0.05), # Anchor point (0.5 = center). Negative y places it below. Adjust y as needed.
                ncol=3, # Adjust columns based on label length and page width
                fontsize=8, title_fontsize=9,
                frameon=False # Optional: remove legend frame
                )

# Adjust layout to prevent labels/titles overlapping and make space for legend below
# Increase bottom margin significantly to accommodate legend
# Fine-tune margins as needed for your specific figure size/aspect ratio
plt.subplots_adjust(bottom=0.25, top=0.90, left=0.08, right=0.98, hspace=0.4, wspace=0.3)

# Display the plot
plt.show()

# Optional: Save the figure with appropriate settings for LaTeX integration
plot_filename_pdf = os.path.join(plot_dir, "model_comparison_metrics_temp0.pdf")
plot_filename_png = os.path.join(plot_dir, "model_comparison_metrics_temp0.png")
try:
    # Use bbox_inches='tight' to automatically adjust boundaries to include legend
    g.figure.savefig(plot_filename_pdf, bbox_inches='tight')
    g.figure.savefig(plot_filename_png, bbox_inches='tight', dpi=300)
    print(f"Saved plot to: {plot_filename_pdf}")
    print(f"Saved plot to: {plot_filename_png}")
except Exception as e:
    print(f"Error saving plot: {e}")

# Melt the DataFrame (using the alphabetically sorted version for consistent plot order)
df_melted = temp0.melt(id_vars='Modello', var_name='Metric', value_name='Value')

# Get the sorted list of models for hue order
sorted_models = temp0['Modello'].unique() # Already sorted alphabetically

# --- Plotting ---
# Set plot style and context
sns.set_theme(style="whitegrid", context="paper")

# Define a palette for the models using the helper function
model_palette = create_family_palette(sorted_models)

# Create a single figure and axes
plt.figure(figsize=(10, 7)) # Adjust figure size as needed (maybe wider now)
ax = sns.barplot(
    data=df_melted,
    x="Metric",          # Metrics on x-axis
    y="Value",           # Score on y-axis
    hue="Modello",       # Group bars by Model
    order=metrics,       # Use the predefined metrics list for x-axis order
    hue_order=sorted_models, # Ensure consistent hue order
    palette=model_palette  # Use palette defined for models
)

# --- Customization ---
# Set plot title and labels
ax.set_title('Model Performance Comparison Across Metrics (Temp 0.0)', fontsize=14, pad=20)
ax.set_xlabel("Metric", fontsize=10)
ax.set_ylabel("Score", fontsize=10)

# Adjust x-axis labels (no rotation needed usually for metrics)
plt.xticks(fontsize=9)
plt.yticks(fontsize=8)

# Set y-axis limits
ax.set(ylim=(0, 1.05))

# Improve legend placement (may need more space now)
# Place legend outside the plot to the right
ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=8, title_fontsize=9)

# Adjust layout to prevent labels/legend overlapping
# Give more space on the right for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust right margin (e.g., 0.85)

# Display the plot
plt.show()
