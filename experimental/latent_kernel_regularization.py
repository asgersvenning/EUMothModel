# Check if global "cls_matrix" exists
if "cls_matrix" not in globals():
  raise NameError("Global variable cls_matrix not found. Please define it in the global scope of the script.")

import torch

def normalized_frobenius_norm(m):
    assert m.shape[0] == m.shape[1], "Input matrices must be square."

    n = m.shape[0]

    # Calculate normalization factor under the assumption that all elements of
    # m are in [-1, 1]
    normalization_factor = ((n ** 2 - n) ** (1/2))

    return torch.norm(m, p = "fro") / normalization_factor

def apdis(activations, gamma = 3):
  dmat = torch.cdist(activations, activations, p = 2)

  return 1 - torch.exp(-dmat / gamma)

def class_structure_loss(activations, labels, return_value = True):
  device, dtype = activations.device, activations.dtype

  # Calculate pairwise dissimilarity between activations of items in the batch
  activation_matrix = apdis(activations)

  # Fetch the precomputed expected pairwise similarity between activations
  # based on the ground-truth labels
  expected_matrix = create_expected_matrix(labels, bias = 0).to(device, dtype)

  # Return the Frobenius norm of the difference between the observed and
  # expected pairwise dissimilarity matrices, normalized to be in [0, 1]
  if return_value:
    return normalized_frobenius_norm(activation_matrix - expected_matrix)
  else:
    return expected_matrix, activation_matrix

def create_symmetric(n, v = None, dtype=torch.float32, bias = 0):
  m = torch.zeros((n,n), dtype = dtype)
  if v is None:
    v = torch.rand((n ** 2 + n) // 2)

  i, j = torch.triu_indices(*m.shape, 0)
  m[i, j] = v
  m.T[i, j] = v
  if bias != 0:
    m += bias
    m /= 1 + bias
  m = m.fill_diagonal_(0)

  return m

def create_expected_matrix(cls, bias = 0):
  global cls_matrix
  if not isinstance(cls, torch.Tensor):
    cls = torch.tensor(cls).int()

  # Create meshgrid for advanced indexing
  c_row, c_col = torch.meshgrid(cls, cls, indexing = "ij")

  m = cls_matrix[c_row, c_col]
  m.requires_grad = False

  if bias != 0:
    m += bias
    m /= 1 + bias
    m.fill_diagonal_(0)
  elif bias < 0:
    raise ValueError(f"Bias must be positive not {bias}")

  return m