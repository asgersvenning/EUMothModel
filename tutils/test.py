import torch
import numpy as np
import matplotlib.pyplot as plt
import math

import sys, os
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tutils.models import parse_state_file

# Create dummy masks for testing purposes
def create_dummy_masks(ns, do_seq=True):
    """
    Creates dummy masks for testing purposes.

    The links between the classes are determined as follows:
    Each class at level i is assigned to a single parent class at level i + 1, such that the parent classes are evenly distributed over the classes at level i.
    The parent classes are assigned sequentially, such that first the number of classes which should be assigned to each parent class, n, is determined.
    Then the first n classes are assigned to the first parent class, the next n classes to the second parent class, etc.

    Args:
        ns (list of int): The number of classes at each level
        do_seq (bool): Whether to assign the parent classes sequentially or randomly
    """
    ms = []
    for i in range(len(ns) - 1):
        m = torch.zeros((ns[i + 1], ns[i]))
        n = math.ceil(ns[i] / ns[i + 1])
        for j in range(ns[i]):
            # Assign the parent class to the class at level i
            if do_seq:
                m[j // n, j] = 1
            else:
                m[np.random.randint(ns[i + 1]), j] = 1
        ms.append(m)
    return ms

def sort_masks(ms):
    """
    Each mask is a matrix of shape (n_classes[level + 1], n_classes[level]) which maps the classes of level to the classes of level + 1 with ones where the classes are linked and zeros where they are not.
    A class at level only has one parent class at level + 1.
    As such mask[i].T @ mask[i] creates a matrix of shape (n_classes[level], n_classes[level]) which encodes which classes of level are linked together at level + 1 (share parent class)

    The real masks are not sorted by level, like the constructed dummy masks are.
    This functions uses Topological Sorting to sort the masks by level, starting at the highest level (the last mask) and ending at the lowest level (the first mask).
    
    Args:
        ms (list of torch.Tensor): The masks to sort

    Returns:
        list of torch.Tensor: The sorted masks and the new order of the classes at each level
    """
    new_order = [torch.arange(m.shape[1]) for m in ms]  # Initialize with original order

    for level in range(len(ms) - 1, -1, -1):
        parents = torch.argmax(ms[level], dim=0)
        parent_order = torch.argsort(parents)

        # Mask sorting
        ms[level] = ms[level][:, parent_order]
        if level > 0:
            ms[level - 1] = ms[level - 1][parent_order, :]

        # Index sorting
        assert len(new_order[level]) == len(parent_order), f"Level {level}: {new_order[level].shape} != {parent_order.shape}"
        new_order[level] = new_order[level][parent_order]

    return ms, new_order

def link_masks(ms, i, j):
    """
    Each mask is a matrix of shape (n_classes[level + 1], n_classes[level]) which maps the classes of level to the classes of level + 1 with ones where the classes are linked and zeros where they are not.
    A class at level only has one parent class at level + 1.
    As such mask[i].T @ mask[i] creates a matrix of shape (n_classes[level], n_classes[level]) which encodes which classes of level are linked together at level + 1 (share parent class)

    In general the recursive formula for creating a mask of shape (n_classes[level], n_classes[level + x]) which encodes the parent classes of level at level + x is:
    link_mask_{i, j} = link_mask_{i, j - 1}.T @ mask[j].T
    link_mask_{i, i} = mask[i].T
    
    Args:
        masks (list of torch.Tensor): The masks to link
        i (int): The level of the first mask
        j (int): The level of the last mask

    Returns:
        torch.Tensor: The linked masks of shape (n_classes[i], n_classes[j])
    """
    if i > j:
        raise ValueError(f"i ({i}) must be smaller than or equal to j ({j})")
    if i < 0 or j >= len(ms):
        raise ValueError(f"i ({i}) and j ({j}) must be in the range [0, {len(ms) - 1}]")
    # Base case
    if i == j:
        bm = ms[i].T
        return bm
    return link_masks(ms, i, j - 1) @ ms[j].T

def reordered_distance_matrix(ms, sort=True):
    """
    This function takes a list of masks and creates a distance matrix between all classes at level 0,
    where the distance is defined as the level at which the classes share a parent class.

    The distance matrix is reordered such that distance[i, j] < distance[i, k] for all k > j and j > i; i.e. distances closer to the diagonal are smaller.

    Args:
        ms (list of torch.Tensor): The masks to link

    Returns:
        torch.Tensor: The reordered distance matrix of shape (n_classes[0], n_classes[0])
        torch.Tensor: The new order of the classes at all levels
        list of torch.Tensor: The reordered masks 
    """
    ms = [m.clone() for m in ms]
    if sort:
        ms, new_order = sort_masks(ms)
    else:
        new_order = [torch.arange(m.shape[1]) for m in ms]
    # Create a distance matrix between all classes at level 0, where the distance is defined as the level at which the classes share a parent class
    links = [link_masks(ms, 0, i) for i in range(len(ms))]
    shared_link = [l @ l.T for l in links]
    shared_levels = torch.stack(shared_link).sum(dim=0)
    shared_levels[torch.eye(shared_levels.shape[0]).bool()] += 1
    distance_matrix = shared_levels.max() - shared_levels
    # Return the reordered distance matrix, the new order of the classes at all levels and the reordered masks
    return distance_matrix, new_order, ms

# Convert logarithmic mask to linear mask
state = parse_state_file("models/empty.state")
masks = [(m > -1).float() for m in state["masks"]]
old_masks = deepcopy(masks)


# Create a color map for the distance matrix
colors = plt.cm.get_cmap("viridis", len(masks) + 2)

# Plot the distance matrix for nodes at level 0
dmat, updated_order, updated_masks = reordered_distance_matrix(masks)
plt.figure(figsize=(5, 5))
plt.imshow(dmat.cpu().float().numpy(), cmap=colors)
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, fc=colors(i)) for i in range(len(masks) + 2)], [str(i) for i in range(len(masks) + 2)],
    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.
)
plt.savefig("plots/links.png", bbox_inches="tight", dpi=400)
plt.close()

dmat, _, _ = reordered_distance_matrix(old_masks, False)
plt.figure(figsize=(5, 5))
plt.imshow(dmat.cpu().float().numpy(), cmap=colors, interpolation="nearest")
# Create a legend for the heatmap which shows which color corresponds to which number, show the legend outside of the heatmap
plt.legend(
    [plt.Rectangle((0, 0), 1, 1, fc=colors(i)) for i in range(len(masks) + 2)], [str(i) for i in range(len(masks) + 2)],
    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.
)
plt.savefig("plots/links_ref.png", bbox_inches="tight", dpi=400)

updated_masks, updated_order = sort_masks(deepcopy(old_masks))
for level in range(len(old_masks)):
    assert torch.allclose(old_masks[level], masks[level]), f"Level {level}: old_masks != masks"
    print(f"Level {level}")
    for old_node in range(old_masks[level].shape[1]):
        old_parent = torch.argmax(old_masks[level], dim=0)[old_node]
        new_node = updated_order[level][old_node]
        new_parent = torch.argmax(updated_masks[level], dim=0)[new_node]

        if level < len(old_masks) - 1:
            old_parent = updated_order[level + 1][old_parent]

        assert old_parent == new_parent, f"Level {level}, Node {old_node}: old_parent ({old_parent}) != new_parent ({new_parent})"
