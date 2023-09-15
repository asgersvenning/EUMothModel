# This script defines the dataloader for the pretraining dataset.
# The data is contained in a remote server, which is mounted to the local filesystem using the mount script.
# The dataset has an inherent hierarchical structure, which is reflected in the directory structure.
# The hierarchy is as follows:
# root
# --Family1
#   --Genus1
#       --Species1
#           img1.jpg/jpeg
#           img2.jpg/jpeg
#           ...
#       --Species2
#           img1.jpg/jpeg
#           img2.jpg/jpeg
#           ...
#       ...
#   --Genus2
#      ...
# --Family2
#   ...

# The classes are the species in the dataset.
# The class labels are the indices of the species, sorted alphabetically, in the dataset.

import os, glob, random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# TODO: add configs for image size, batch size, etc.
# config = get_config()['pretraining']

transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])