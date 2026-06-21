import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/Users/karm1616/Desktop/Univeristy/Masters/Machine Learning for the Quanitfied Self/ML4QS-Bouldering/classicalMachineLearningCode/UPDATED_FEATURES_B3/bouldering_summary_1.csv")

difficulty_counts = df["difficulty"].value_counts().sort_index()
topped_counts = df["topped"].value_counts().sort_index()
style_counts = df["style"].value_counts().sort_index()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

datasets = [
    (difficulty_counts, "Boulder Difficulty", "Difficulty", ["Easy", "Medium", "Hard"]),
    (topped_counts, "Completed Attempts", "Topped", ["Not Topped", "Topped"]),
    (style_counts, "Boulder Style", "Style", ["Normal", "Overhang", "Slab", "Dyno"]),
]

for ax, (counts, title, xlabel, label_names) in zip(axes, datasets):
    values = list(counts.values)

    # Use fixed positions: 0, 1, 2, ...
    x_positions = range(len(values))

    ax.bar(x_positions, values)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Attempts")

    # Match labels exactly to the bar positions
    ax.set_xticks(x_positions)
    ax.set_xticklabels(label_names, rotation=30, ha="right")

    # Add values above bars
    for x, value in zip(x_positions, values):
        ax.text(x, value + 0.3, str(value), ha="center")

plt.tight_layout()
plt.show()