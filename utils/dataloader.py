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

import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import utils.config as c

# TODO: add configs for image size, batch size, etc.
config = c.get_dataloader_config()
root_dir = c.get_mount_config()['local']

# All images have a unique id, which is used in their naming scheme: UUID.jpg/jpeg
class PretrainingImages():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []
        self.family = []
        self.genus = []
        self.species = []
        self.uuids = []

        for family in os.listdir(root_dir):
            for genus in os.listdir(os.path.join(root_dir, family)):
                for species in os.listdir(os.path.join(root_dir, family, genus)):
                    for image in os.listdir(os.path.join(root_dir, family, genus, species)):
                        uuid = image.split('.')[0]
                        self.family.append(family)
                        self.genus.append(genus)
                        self.species.append(species)
                        self.uuids.append(uuid)
                        self.images.append(os.path.join(root_dir, family, genus, species, image))

    def __len__(self):
        return len(self.uuids)
    
    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("Expected int, got {}".format(type(idx)))
        if idx < 0 or idx >= len(self.uuids):
            raise IndexError("Index out of bounds")
        
        return self.images[idx], self.family[idx], self.genus[idx], self.species[idx], self.uuids[idx]


class PretrainingDataset(Dataset):
    def __init___(self, root_dir=root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = PretrainingImages(root_dir)
        self.classes = sorted(list(set(self.images.species)))
        self.class_labels = {self.classes[i] : i for i in range(len(self.classes))}
        self.label2class = {i : self.classes[i] for i in range(len(self.classes))}
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path, family, genus, species, uuid = self.images[idx]

        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        
        return image, self.class_labels[species]