import torch
import torch.nn.functional as F
import numpy as np

from collections import Counter

def class_counting(paths, class_handles, class_parser = lambda x: x.split("/")[-2:-5:-1]):
    """
    Count the number of each class in the labels
    """
    counts = [Counter(range(l)) for l in class_handles["n_classes"]]
    for path in paths:
        components = class_parser(path)
        for ctype, class_str in enumerate(components):
            counts[ctype][class_handles["class_to_idx"][ctype][class_str]] += 1

    # Convert to NumPy arrays
    counts = [np.array(list(counter.values())) for counter in counts]

    return counts

def create_hierarchy(paths, class_handles, class_parser = lambda x: x.split("/")[-2:-5:-1]):
    """
    Creates a hierarchy from the paths and class handles.
    The hierarchy is constructed based on the nodes found in the dataset. 
    TODO: The hierarchy should be constructed once and saved in a structured file.

    Arguments:
        paths (list): List of paths to the images in the dataset.
        class_handles (dict): Dictionary containing the class handles. It should contain the keys "n_classes" and "class_to_idx". 
            Each of these should be a list of length equal to the number of layers in the hierarchy. 
            "n_classes" should contain the number of classes at each level, and "class_to_idx" should contain a dictionary mapping the class strings to indices at each level.

    Returns:
        list: A list for each level of the hierarchy. Each list contains a list for each node containing the indices of the children of that node. 
            Level 0 is the leaf level, and is not included. 
    """
    hierarchy = [[[]] * n for i, n in enumerate(class_handles["n_classes"]) if i != 0]  # Create empty lists for each level
    processed_leaves = [0] * class_handles["n_classes"][0]  # Keep track of which leaves have been processed

    # Iterate over the paths (labels are embedded as subdirectories in the path)
    for path in paths:
        # Split the path into its components and reverse the order (This assumes the structure of the subdirectories) TODO: Make this more general
        components = class_parser(path)
        # Convert the class strings to indices
        indices = [class_handles["class_to_idx"][ctype][class_str] for ctype, class_str in enumerate(components)] 
        
        # Skip processed leaves (species in this case)
        if processed_leaves[indices[0]] == 0:  # If the leaf has not been processed yet
            processed_leaves[indices[0]] = 1
        else:
            continue  # Skip this leaf
        
        # Iterate over the indices and add them to the hierarchy
        for i in range(1, len(indices)):
            # TODO: This is a hack to make the indices line up with the hierarchy levels; 
            # this should be moved to the range() function, but this would require all 'i's in the loop to be changed
            i -= 1 
            # Get the parent and child indices
            parent = indices[i + 1]
            child = indices[i]
            
            # Create a list for the parent if it does not exist 
            # (this is done to avoid the lists being shallow copies of each other, i.e. the same list, which creates incorrect hierarchies)
            if not hierarchy[i][parent]:
                hierarchy[i][parent] = [child]
            # Otherwise, append the child to the list
            else:
                hierarchy[i][parent].append(child)  # Append the child to the parent's list
    
    # Ensure that the lists of children contain no duplicates. 
    # TODO: This is inefficient, and should be done in the loop above
    for i in range(1, len(hierarchy)):
        for j in range(len(hierarchy[i])):
            hierarchy[i][j] = list(set(hierarchy[i][j]))

    return hierarchy

def create_mask(indices, s, zero=-100, **kwargs):
    """
    Create an approximate logarithmic binary mask with the given indices.

    Arguments:
        indices (list): List of indices to include in the mask.
        s (int): Size of the mask.
        zero (int): "Approximate zero" value. This is used to avoid numerical issues with log(0). 
            This should be a large negative number. Default: -100.
        **kwargs: Keyword arguments to pass to torch.zeros(). Notably 'device' and 'dtype'.
    
    Returns:
        torch.Tensor: An approximate logarithmic binary mask for the given indices.
    """
    t = torch.zeros(s, **kwargs, requires_grad=False)
    t += zero
    t[indices] = 0
    return t

def mask_hierarchy(hierarchy, zero=-100, **kwargs):
    """
    Create approximate logarithmic binary masks for the given hierarchy.

    Arguments:
        hierarchy (list): List of lists of lists of indices. 
            The first level of the list corresponds to the levels of the hierarchy, and each level contains a list of lists of indices for each node.
        zero (int): "Approximate zero" value. This is used to avoid numerical issues with log(0).
        **kwargs: Keyword arguments to pass to torch.zeros(). Notably 'device' and 'dtype'.

    Returns:
        list: List of masks for each level of the hierarchy.
            Each mask has shape (n_nodes, n_child_nodes) and can be used to calculate the logits for the nodes based on the child logits:
            TODO: Add equation here (logarithmic matrix multiplication)
    """
    masks = []
    for level in hierarchy:
        n = sum([len(indices) for indices in level])
        masks.append([create_mask(indices, n, zero=zero, **kwargs) for indices in level])

    return [torch.stack(level) for level in masks]

def create_random_P(n_classes, **kwargs):
    """
    Create a random P matrix for testing with the given number of classes at each level.
    """
    raise NotImplementedError("This function is not implemented properly yet.")
    P = [None] * len(n_classes)
    P[0] = torch.rand(n_classes[0], **kwargs)  # Create a random P matrix for the leaf level
    P[0] = F.log_softmax(P[0], dim=-1)  # Normalize along class dimension
    for i, n in enumerate(n_classes):
        if i == 0:
            continue
        P[i] = P[i-1] * masks[i-1]  # Multiply by the mask of the previous level
        P[i] = torch.logsumexp(P[i], dim=-1)

    return P