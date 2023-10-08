# This script aims to somewhat balance the classes in the dataset, by increasing the entropy of the class distribution.
# The approach that will be taken will "slide" a vertical bar down the class distribution histogram, until X% of the area is under the bar.

# %%
# Import modules
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Move working directory to the root of the project
os.chdir("/home/ucloud/EUMothModel")
print("Working directory:", os.getcwd())
sys.path.append(os.getcwd())

# Import custom modules for ERDA data transfer
from utils.implicit_mount import *
from utils.dataloader import *


if __name__ == "__main__":
    # %%

    # Load dataset
    backend = IOHandler(verbose = False, clean=True)
    backend.start()
    backend.cd("AMI_GBIF_Pretraining_Data/without_larvae")
    backend.cache_file_index(skip=0)

    # %%

    # Get class distribution
    paths = backend.cache["file_index"]
    classes = [i.split("/")[-2] for i in paths]
    unique_classes, counts = np.unique(classes, return_counts=True)
    class_distribution = {c: n for n, c in sorted(zip(counts, unique_classes), reverse=True)}
    print("Class distribution:", class_distribution)

    # %%

    # Plot class distribution
    plt.figure(figsize=(20, 10))
    plt.bar(class_distribution.keys(), class_distribution.values())
    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    # Logarithmic scale on y-axis
    plt.yscale("log")
    # Remove xticks/labels
    plt.xticks([])
    # Save plot
    plt.savefig("logs/class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    backend.stop()