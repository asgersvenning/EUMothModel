import torch, torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

import sys, os
import warnings
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from class_handling import create_hierarchy, mask_hierarchy
from models import save_state_file, parse_state_file

classes = [["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], ["03689", "147", "25"], ["03689", "12457"]]
class_to_idx = [{str(i) : ind for ind, i in enumerate(c)} for c in classes]
idx_to_class = [{v : k for k, v in c.items()} for c in class_to_idx]
n_classes = [len(c) for c in classes]

nested_classes = []
for i in classes[0]:
    nested_classes.append([])
    for j in classes[1]:
        if not i in j:
            continue
        for k in classes[2]:
            if not i in k:
                continue
            nested_classes[-1] = (i + "/" + j + "/" + k)

hierarchy = create_hierarchy(nested_classes, {"n_classes" : n_classes, "class_to_idx" : class_to_idx}, lambda x: x.split("/"))

masks = mask_hierarchy(hierarchy)

class_handles = {
    "classes" : classes,
    "n_classes" : n_classes,
    "class_to_idx" : class_to_idx,
    "idx_to_class" : idx_to_class,
    "hierarchical" : True
}

state = {
    "class_handles" : class_handles,
    "masks" : masks
}

# save_state_file(state, "models/empty_MNIST.state")

# Setup
preprocessor = lambda x : x / 255.

batch_size = 2
num_workers = 4

def mnist_to_hierarchy(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, list):
        x = [x]
    cls_leaf = [class_handles["idx_to_class"][0][i] for i in x]
    cls_names = [None] * len(cls_leaf)
    for idx, i in enumerate(cls_leaf):
        for j in class_handles["classes"][1]:
            if not i in j:
                continue
            for k in class_handles["classes"][2]:
                if not i in k:
                    continue
                cls_names[idx] = i, j, k
    cls_idx = [class_handles["class_to_idx"][level][name] for tcls in cls_names for level, name in enumerate(tcls)]
    return cls_idx

toT = torchvision.transforms.PILToTensor()
mnist_preprocessor = lambda x : preprocessor(toT(x).repeat(3, 1, 1).float())

# Dataset and dataloader setup
full_dataset = train = torchvision.datasets.MNIST(
    root="data", 
    train=True, 
    download=True, 
    transform=mnist_preprocessor,
    target_transform=mnist_to_hierarchy
)
full_dataset.class_handles = deepcopy(parse_state_file("models/empty_MNIST.state")["class_handles"])

datasets = torch.utils.data.random_split(full_dataset, [int(0.95 * len(full_dataset)), int(0.01 * len(full_dataset)), len(full_dataset) - int(0.95 * len(full_dataset)) - int(0.01 * len(full_dataset))])
dtrain, dval, dtest = datasets
train = torch.utils.data.DataLoader(dtrain, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val = torch.utils.data.DataLoader(dval, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=True, num_workers=num_workers)

train_cls_idx = full_dataset.targets[dtrain.indices]

def mnist_class_counts(targets, target_transform):
    from collections import Counter
    counts = [Counter(range(i)) for i in class_handles["n_classes"]]
    for target in targets:
        for ctype, class_idx in enumerate(target_transform(target)):
            counts[ctype][class_idx] += 1
    counts = [np.array(list(counter.values())) for counter in counts]
    return counts

train_counts = mnist_class_counts(train_cls_idx, mnist_to_hierarchy)

# Test train loader
for i, (x, y) in enumerate(train):
    print(x[0][0].shape)
    print(x[0][0])
    print(y)
    break