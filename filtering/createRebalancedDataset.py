# This script is used to prune images from high-frequency classes in the dataset, in order to rebalance the class distribution.

# `os` is imported first to ensure that the working directory is set to the root of the project
import os, sys

# Move working directory to the root of the project
os.chdir("/home/ucloud/EUMothModel")
print("Working directory:", os.getcwd())
sys.path.append(os.getcwd())

# Import custom modules for ERDA data transfer
from utils.implicit_mount import *
from utils.dataloader import *

# Import other modules
from tqdm import tqdm
import numpy as np

## Helper functions
# Remote file copying helper function
def move_to_dir(paths, dst_dir, client: "IOHandler", P=10, verbose=False):
    # Template: mget -d -O sftp://io.erda.au.dk:/AMI_GBIF_Pretraining_Data/NEW_SUPER_DIRECTORY ./PATH_TO_IMAGES
    # The template above is used to copy a file from one location on the remote server to another location on the remote server,
    # while preserving the directory structure of the source location and automatically creating the necessary subdirectories in the destination location.

    cmd_prefix = f"mget -d -P {P} -O sftp://io.erda.au.dk:/AMI_GBIF_Pretraining_Data/{dst_dir}"

    cmd_suffix = []
    for path in paths:
        cmd_suffix.append(f"./{path}")
    
    cmd_suffix = " ".join(cmd_suffix)

    cmd = f"{cmd_prefix} {cmd_suffix}"
    if verbose:
        print("Executing command:", cmd)
    result = client.execute_command(cmd, blocking=True, execute=True)
    if verbose:
        print("Result:", result)

class ClassDistribution(dict):
    """An ordered dictionary that can be used to represent a class distribution."""
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

    ## Hyperparameters
    skip = 0

    # ERDA data transfer setup
    backend = None
    if backend is not None:
        backend.stop()
    backend = IOHandler(verbose = False, clean=True)
    backend.start()
    backend.cd("AMI_GBIF_Pretraining_Data/without_larvae")
    backend.cache_file_index(skip=skip)

    classes = [i.split("/")[-2] for i in backend.cache["file_index"]]
    unique_classes, counts = np.unique(classes, return_counts=True)
    class_distribution = ClassDistribution(unique_classes, counts)

    # Find cutoff point for high-frequency classes
    cutoff = slice_distribution(class_distribution, 0.75)

    # Get pruned path list
    pruned_paths = []
    class_pruned_counts = {cls : 0 for cls in class_distribution.classes}
    for path in tqdm(backend.cache["file_index"], desc="Pruning paths"):
        cls = path.split("/")[-2]
        if class_pruned_counts[cls] < cutoff:
            pruned_paths.append(path)
            class_pruned_counts[cls] += 1

    # Split paths by class
    class_paths = {cls : [] for cls in class_distribution.classes}
    for path in tqdm(pruned_paths, desc="Splitting paths by class"):
        cls = path.split("/")[-2]
        class_paths[cls].append(path)

    # with open("logs/test_rebalanced50_without_larvae.txt", "w") as f:
    #     # f.writelines(class_paths[list(class_paths.keys())[0]])
    #     # f.write("\n")
    #     for cls, paths in class_paths.items():
    #         f.write(f"{cls}: {len(paths)}\n")
    
    # Copy pruned paths to new directory
    copy_pbar = tqdm(class_paths.items(), desc="Copying pruned paths", dynamic_ncols=True)
    for cls, paths in copy_pbar:
        copy_pbar.set_description(f"Copying pruned paths for class {cls} ({len(paths)} paths)")
        move_to_dir(paths, f"rebalanced50_without_larvae", backend, verbose=False)

    # Close backend
    backend.stop()