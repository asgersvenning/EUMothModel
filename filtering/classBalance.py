# This script aims to somewhat balance the classes in the dataset, by increasing the entropy of the class distribution.
# The approach that will be taken will "slide" a vertical bar down the class distribution histogram, until X% of the area is under the bar.

# Import standard libraries
import os, sys
import numpy as np

# Import third-party libraries
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# Move working directory to the root of the project
os.chdir("/home/ucloud/EUMothModel")
print("Working directory:", os.getcwd())
sys.path.append(os.getcwd())

# Custom modules
from utils.implicit_mount import *
from utils.dataloader import *

# Set monospace as the default font for this script
plt.rcParams['font.family'] = 'monospace'

class ClassDistribution(dict):
    ### An ordered dictionary that can be used to represent a class distribution.
    def __init__(self, keys, values):
        _, keys, values = zip(*sorted(zip(values, keys, values), reverse=True))
        super().__init__(zip(keys, values))

    def size(self):
        return sum(self.counts)
    
    def __repr__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.items()])

    @property
    def counts(self):
        return list(self.values())
    
    @property
    def frequencies(self):
        s = self.size()
        return [v / s for v in self.counts]
    
    @property
    def classes(self):
        return list(self.keys())


def slice_distribution(class_distribution, proportion):
    """Find the y-value that splits the class distribution by the given proportion."""
    target = sum(class_distribution.counts) * proportion
    y_min = 0
    y_max = max(class_distribution.counts)
    y = (y_min + y_max) / 2

    while True:
        area = sum([min(y, v) for v in class_distribution.counts])
        if abs(area - target) < 1:
            break
        elif area < target:
            y_min = y
        else:
            y_max = y
        ran = y_max - y_min
        if ran < 1e-9:
            break
        y = (y_min + y_max) / 2
    return y

if __name__ == "__main__":
    # Initialize IOHandler and load dataset
    backend = IOHandler(verbose=False, clean=True)
    backend.start()
    backend.cd("AMI_GBIF_Pretraining_Data/without_larvae")
    backend.cache_file_index(skip=0)

    # Get class distribution
    paths = backend.cache["file_index"]
    classes = [i.split("/")[-2] for i in paths]
    unique_classes, counts = np.unique(classes, return_counts=True)
    class_distribution = ClassDistribution(unique_classes, counts)

    # Configure the plot
    plt.figure(figsize=(20, 10))
    
    # Create the Seaborn barplot with a different color (let's say gray)
    plt.bar(range(len(class_distribution)), class_distribution.counts, color='gray', edgecolor='gray', linewidth=0)

    # Additional aesthetics and color mapping
    colors = plt.cm.viridis(np.linspace(0, 1, 11))

    # Pre-calculate maximum y and n values for text field width
    ps = np.arange(0.1, 1.1, 0.1)
    ys = [slice_distribution(class_distribution, p) for p in ps]
    min_y = min(ys)
    max_y = max(ys)
    max_n = sum(class_distribution.counts)
    field_width_y = len(str(int(max_y)))
    field_width_n = len(str(int(max_n)))
    right_limit   = len(class_distribution)

    # Add lines and annotations
    for i, (y, p) in enumerate(zip(ys, ps)):
        n = sum([min(y, v) for v in class_distribution.counts])
        c = colors[i]

        plt.axhline(y=y, color=c, linestyle="--", label=f"{p*100}%")
        
        # Left-anchored text, with some padding to avoid overlap
        plt.text(-75, y * 1.1, f"{p*100:.0f}%", color=c, fontsize=12)  # Adjusted the x-position to -200 for padding

        y_str = f"{y:>{field_width_y}.0f}"
        n_str = f"{n:>{field_width_n}.0f}"
        aligned_text = f"{y_str} ⟶ {n_str}"

        plt.text(right_limit - 0.025 * right_limit, y * 1.1, aligned_text, color=c, fontsize=12, ha='right')

    # Adjust x-axis limits to make space for the left-anchored text
    plt.xlim([-100, right_limit])  # Added padding to the left side of x-axis

    # Additional plot settings
    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.xticks([])

    # Save the figure
    plt.savefig("logs/class_distribution.png", dpi=300, bbox_inches="tight")

    # Close the plot to free up resources
    plt.close()

    percs = np.arange(0.001, 1.001, 0.001)
    sh_evenness = []
    kl_div = []
    ys = []
    for p in percs:
        y = slice_distribution(class_distribution, p)
        tc = [min(y, v) for v in class_distribution.counts]
        tc = np.array(tc)
        tc = tc / sum(tc)
        tsh_entr = -sum(tc * np.log(tc))
        tsh_even = (tsh_entr / np.log(len(tc)))
        tkl_div = np.log(len(tc)) - tsh_entr
        sh_evenness.append(tsh_even)
        kl_div.append(tkl_div)
        ys.append(y)

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot the second line and set properties
    sns.lineplot(x=1-percs, y=ys, ax=ax1, color="forestgreen", label='Max samples per class')
    ax1.set_ylabel("Max samples per class")

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the first line and set properties
    sns.lineplot(x=1-percs, y=kl_div, ax=ax2, color="royalblue", label='Shannon Unevenness')
    ax2.set_xlabel("Proportion of data removed")
    ax2.set_ylabel("Shannon Unevenness")

    # Set x-ticks
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_xticklabels([f"{i*100:.0f}%" for i in np.arange(0, 1.1, 0.1)])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max(ys)*1.01)
    ax2.set_ylim(0, max(kl_div)*1.01)

    # Set tick params
    ax1.tick_params(axis='y', colors="forestgreen")
    ax2.tick_params(axis='y', colors="royalblue")

    # Set panel borders by directly modifying the Axes objects
    ax2.spines['left'].set_color('forestgreen')
    ax2.spines['right'].set_color('royalblue')

    # Create a combined legend
    forestgreen_line = mlines.Line2D([], [], color='forestgreen', label='Max samples per class')
    royalblue_line = mlines.Line2D([], [], color='royalblue', label='KL Div. from Unif.')
    # Remove legend from ax1
    ax1.legend([], [], frameon=False)
    # Add legend to ax2
    ax2.legend(handles=[forestgreen_line, royalblue_line], loc='upper right')

    # Save the figure
    fig.savefig("logs/combined_plot.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    


    # Stop the backend
    backend.stop()