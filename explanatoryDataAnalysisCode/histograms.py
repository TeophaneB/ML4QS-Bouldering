import matplotlib.pyplot as plt

# Looks at bouldering_summary_1.csv and creates histograms for the categorical variables: difficulty, topped, and style.
import pandas as pd 

# Load the dataset
df = pd.read_csv("FEATURES/bouldering_summary_1.csv")
# Count the occurrences of each category in the relevant columns
difficulty_counts = df["difficulty"].value_counts().sort_index()
topped_counts = df["topped"].value_counts().sort_index()
style_counts = df["style"].value_counts().sort_index()

# Create one figure with 3 plots next to each other
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
# Plot histograms for each categorical variable
datasets = [
    (difficulty_counts, "Boulder Difficulty", "Difficulty"),
    (topped_counts, "Completed Attempts", "Topped"),
    (style_counts, "Boulder Style", "Style")
]
# Loop through each dataset and corresponding axis to create the bar plots
for ax, (counts, title, xlabel) in zip(axes, datasets):
    categories = list(counts.keys())
    values = list(counts.values)

    ax.bar(categories, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Attempts")

    # Add values above bars
    for i, value in enumerate(values):
        ax.text(i, value + 0.3, str(value), ha="center")

    # Rotate labels slightly if needed
    ax.tick_params(axis="x", rotation=30)

plt.suptitle("Distribution of Batch 1 Bouldering Attempts", fontsize=14)
plt.tight_layout()

# Save as one image
plt.savefig("class_distributions(RECENT).png", dpi=300, bbox_inches="tight")
plt.show()