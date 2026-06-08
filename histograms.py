import matplotlib.pyplot as plt

# Data
difficulty_counts = {
    "Easy": 9,
    "Medium": 18,
    "Hard": 8
}

topped_counts = {
    "Yes": 26,
    "No": 9
}

style_counts = {
    "Normal": 19,
    "Slab": 5,
    "Overhang": 8,
    "Dynamic": 2
}

# Create one figure with 3 plots next to each other
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

datasets = [
    (difficulty_counts, "Boulder Difficulty", "Difficulty"),
    (topped_counts, "Completed Attempts", "Topped"),
    (style_counts, "Boulder Style", "Style")
]

for ax, (counts, title, xlabel) in zip(axes, datasets):
    categories = list(counts.keys())
    values = list(counts.values())

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
plt.savefig("batch1_categorical_distributions.png", dpi=300, bbox_inches="tight")
plt.show()