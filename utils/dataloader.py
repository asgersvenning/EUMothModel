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

import os, re
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import utils.config as c

# TODO: add configs for image size, batch size, etc.
config = c.get_dataloader_config()
root_dir = c.get_mount_config()['local']

# All images have a unique id, which is used in their naming scheme: UUID.jpg/jpeg
class PretrainingImages:
    def __init__(self, dir, index = True, n_max = -1):
        self.dir = dir
        self.images = []
        self.family = []
        self.genus = []
        self.species = []
        self.uuids = []

        if index:
            index_file = f'{self.dir}{os.sep}folder_index.txt'
            if not os.path.exists(index_file):
                raise ValueError(f'Index file {index_file} does not exist')
            with open(index_file, "r") as index:
                for line_i, path in enumerate(index.readlines()):
                    if n_max != -1 and line_i >= n_max:
                        break
                    path = path.rstrip('\n')
                    file_ext = re.findall("\\.\w{2,4}$", path)
                    if not file_ext or len(file_ext) > 1:
                        print(f'No or invalid file extension found for {path}')
                        continue
                    file_ext = file_ext[0]
                    self.images += [path]
                    parts = re.findall(f'(?<={re.escape(self.dir)}{os.sep}).+', path)[0]
                    family, genus, species, uuid = parts.split(os.sep)
                    uuid = uuid.rstrip(file_ext)
                    self.family += [family]
                    self.genus += [genus]
                    self.species += [species]
                    self.uuids += [uuid]
        else:
            assert NotImplementedError("Indexing not implemented yet")
        # for family in os.listdir(root_dir):
        #     for genus in os.listdir(os.path.join(root_dir, family)):
        #         for species in os.listdir(os.path.join(root_dir, family, genus)):
        #             for image in os.listdir(os.path.join(root_dir, family, genus, species)):
        #                 uuid = image.split('.')[0]
        #                 self.family.append(family)
        #                 self.genus.append(genus)
        #                 self.species.append(species)
        #                 self.uuids.append(uuid)
        #                 self.images.append(os.path.join(root_dir, family, genus, species, image))

    def __len__(self):
        return len(self.uuids)
    
    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("Expected int, got {}".format(type(idx)))
        if idx < 0 or idx >= len(self.uuids):
            raise IndexError("Index out of bounds")
        
        return self.images[idx], self.family[idx], self.genus[idx], self.species[idx], self.uuids[idx]


class PretrainingDataset(Dataset):
    def __init__(self, dir=root_dir, n_max=-1, transform=None):
        super().__init__()
        self.dir = dir
        self.transform = transform
        self.images = PretrainingImages(dir=self.dir, n_max=n_max)
        self.classes = sorted(list(set(self.images.species)))
        self.class_labels = {self.classes[i] : i for i in range(len(self.classes))}
        self.label2class = {i : self.classes[i] for i in range(len(self.classes))}
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path, family, genus, species, uuid = self.images[idx]

        image_type = os.path.splitext(image_path)[1]
        if not image_type in ['.jpg', '.jpeg', '.png']:
            image = self.read_non_standard_image(image_path, image_type)
        else:
            image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        
        return image, self.class_labels[species]
    
    def read_non_standard_image(self, image_path, type):
        # TODO: implement this
        # The function should handle diverse image formats, such as .tif and .bmp.
        # The function should return a tensor.
        raise NotImplementedError(f"Reading non-standard images ({type}) not implemented yet!")