import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
from copy import deepcopy
import os
import zipfile

from typing import Union, Tuple, List

_UNSUPPORTED_MODELS = [
    'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 
    'fcos_resnet50_fpn', 
    'keypointrcnn_resnet50_fpn', 
    'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 
    'mvit_v1_b', 'mvit_v2_s', 
    'raft_large', 'raft_small', 
    'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 
    'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 
    'swin3d_b', 'swin3d_s', 'swin3d_t', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 
    'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32'
    ]

def parse_state_file(path):
    """
    This a helper function to parse the state file of a HierarchicalClassifier.
    The state file is a zip file which contains an optional state_dict (weights - torch .pt file), a class_handles file (pickle), and a masks file (pickle).

    Args:
        path: The path to the state file

    Returns:
        A dictionary containing the state_dict, class_handles, and masks
    """
    state = {}
    # Open the zip file
    with zipfile.ZipFile(path, "r") as zip_file:
        # Check if there is a state_dict
        if "state_dict.pt" in zip_file.namelist():
            # Load the state_dict
            state["state_dict"] = torch.load(zip_file.open("state_dict.pt"), map_location="cpu")
        # Load the class_handles
        state["class_handles"] = pickle.load(zip_file.open("class_handles.pkl"))
        # Load the masks
        state["masks"] = torch.load(zip_file.open("masks.pt"), map_location="cpu")
        # Ensure numeric stability of the masks
        for mi in range(len(state["masks"])):
            state["masks"][mi][state["masks"][mi] < -1] = -100
            state["masks"][mi][state["masks"][mi] >= -1] = 0
    return state

def save_state_file(state, path):
    # Package the state files into a zip file
    with zipfile.ZipFile(path, "w") as zip_file:
        for key, value in state.items():
            if key in ["state_dict", "masks"]:
                with zip_file.open(key + ".pt", "w") as f:
                    torch.save(value, f)
            else:
                zip_file.writestr(key + ".pkl", pickle.dumps(value))


def package_old_files(weights_path : Union[str, None] = None, class_handles_path : Union[str, None] = None, masks_path : Union[str, None] = None, output_path : str = None) -> None:
    """
    This is a helper function to package the old files into a state file.

    Args:
        weights_path: The path to the weights file
        class_handles_path: The path to the class_handles file
        masks_path: The path to the masks file
        output_path: The path to save the state file to; must be specified!

    Returns:
        None
    """
    # Type checking
    if output_path is None:
        raise ValueError("output_path must be specified")
    if not isinstance(output_path, str):
        raise ValueError("output_path must be a string")
    else:
        if os.path.exists(output_path):
            raise ValueError("output_path already exists")
        if os.path.splitext(output_path)[1] != ".state":
            Warning("output_path does not have the canonical .state extension.")
    if weights_path is not None and not isinstance(weights_path, str):
        raise ValueError("weights_path must be a string or None")
    if class_handles_path is not None and not isinstance(class_handles_path, str):
        raise ValueError("class_handles_path must be a string or None")
    if masks_path is not None and not isinstance(masks_path, str):
        raise ValueError("masks_path must be a string or None")
    # Load the state files
    state = {}

    # Helper function to load the files (either .pt or .pkl)
    def load_file(path):
        file_type = os.path.splitext(path)[1]
        if not file_type in [".pt", ".pkl"]:
            raise ValueError(f"File type {file_type} is not supported.")
        if file_type == ".pt":
            return torch.load(path, map_location="cpu")
        elif file_type == ".pkl":
            return pickle.load(open(path, "rb"))
    
    if weights_path is not None:
        state["state_dict"] = load_file(weights_path)
    if class_handles_path is not None:
        state["class_handles"] = load_file(class_handles_path)
    if masks_path is not None:
        state["masks"] = load_file(masks_path)
    
    save_state_file(state, output_path)

class HierachicalPrediction(list):
    def __init__(self, predictions):
        super(HierachicalPrediction, self).__init__(predictions)
    
    def __repr__(self):
        n_pred = len(self[0])
        out = "\n"
        for i in range(n_pred):
            out += f"PREDICTION {i}\n"
            for j in range(len(self)):
                tpl = self[j][i]
                cls, conf = tpl
                out += f"Level {j} : {cls}, Confidence: {100 * conf:.1f}%\n"
        return out
    
    def transpose(self):
        return [list(i) for i in zip(*self)]


class HierarchicalClassifier(nn.Module):
    """
    A custom hierarchical classifier module. This module is used to replace the classifier of an arbitrary PyTorch model.

    Args:
        num_features: The number of features in the input to the classifier
        num_classes: The number of classes in each level of the hierarchy
        masks: A list of masks to use for the hierarchical classification
        device: The device to place the module on
        dtype: The data type to use for the module
    
    Returns:
        A HierarchicalClassifier module
    """
    def __init__(self, num_features, num_classes, masks, device=torch.device("cpu"), dtype=torch.bfloat16):
        super(HierarchicalClassifier, self).__init__()
        self.dropout1 : nn.Dropout = nn.Dropout(0.2)
        self.bn1 : nn.BatchNorm1d = nn.BatchNorm1d(num_features, device=device, dtype=dtype)
        self.linear1 : nn.Linear = nn.Linear(num_features, 1024, device=device, dtype=dtype)
        self.bn2 : nn.BatchNorm1d = nn.BatchNorm1d(1024, device=device, dtype=dtype)
        self.dropout2 : nn.Dropout = nn.Dropout(0.1)
        self.linear2 : nn.Linear = nn.Linear(1024, 1024, device=device, dtype=dtype)
        self.bn3 : nn.BatchNorm1d = nn.BatchNorm1d(1024, device=device, dtype=dtype)
        self.dropout3 : nn.Dropout = nn.Dropout(0.1)
        self.linear3 : nn.Linear = nn.Linear(1024, 512, device=device, dtype=dtype)
        self.bn4 : nn.BatchNorm1d = nn.BatchNorm1d(512, device=device, dtype=dtype)
        self.dropout4 : nn.Dropout = nn.Dropout(0.1)
        self.leaf_logits : nn.Linear = nn.Linear(512, num_classes[0], device=device, dtype=dtype)
        self.bn5 : nn.BatchNorm1d = nn.BatchNorm1d(num_classes[0], device=device, dtype=dtype)
        self.masks : List[torch.Tensor] = [mask.to(device=device, dtype=dtype).requires_grad_(False) * 10 for mask in deepcopy(masks)]
        self.return_embeddings : bool = False
        # self.masks = torch.jit.Attribute([mask.to(device=device, dtype=dtype).requires_grad_(False) for mask in deepcopy(masks)], List[torch.Tensor])
        # self.return_embeddings = torch.jit.Attribute(False, bool)

        self.silu : nn.SiLU = nn.SiLU()
    
    # @torch.compile(fullgraph=True, mode="max-autotune")
    # def forward(self, x) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
    def forward(self, x) -> List[torch.Tensor]:
        if self.return_embeddings:
            embeddings = x.clone()
        x = self.dropout1(x)
        # x = self.bn1(x)
        x = self.linear1(x) # Layer 1
        # x = self.bn2(x)
        x = self.silu(x)
        x = self.dropout2(x)
        x = self.linear2(x) # Layer 2
        # x = self.bn3(x)
        x = self.silu(x)
        x = self.dropout3(x)
        x = self.linear3(x) # Layer 3
        # x = self.bn4(x)
        x = self.silu(x)
        x = self.dropout4(x)
        x = self.leaf_logits(x) # Leaf logits (Layer 4)
        x = self.silu(x)
        # x = self.bn5(x)
        # Compute the normalized log probabilities for the leaf nodes (level 0)
        y0 = F.log_softmax(x, dim = 1)
        ys = [y0]
        # Propagate the probabilities up the hierarchy using the masks
        for mask in self.masks:
            ys.append(torch.logsumexp(ys[-1].unsqueeze(2) + mask.T, dim = 1))
        if self.return_embeddings:
            return ys, embeddings
        else:
            return ys
        
    def to(self, device=None, dtype=None):
        if device is not None:
            self.masks = [mask.to(device=device) for mask in self.masks]
        if dtype is not None:
            self.masks = [mask.to(dtype=dtype) for mask in self.masks]
        return super(HierarchicalClassifier, self).to(device=device, dtype=dtype)
    
    def cuda(self, device=None):
        self.masks = [mask.cuda(device=device) for mask in self.masks]
        return super(HierarchicalClassifier, self).cuda(device=device)

def create_extendable_model_class(backbone_model : Union[str, torch.nn.Module], model_args : dict={}, classifier_name : Union[str, List[str]]=["classifier", "fc"]) -> torch.nn.Module:
    # Resolve the backbone model to a nn.Module
    if isinstance(backbone_model, str):
        if backbone_model in _UNSUPPORTED_MODELS:
            raise ValueError(f"The model {backbone_model} is not supported.")
        backbone_model = torchvision.models.get_model(backbone_model, **model_args)
    elif not isinstance(backbone_model, nn.Module):
        raise ValueError("backbone_model must be a string or a torch.nn.Module")
    
    # Find the name of the classifier module in the backbone model
    backbone_classifier_name = None
    if isinstance(classifier_name, str):
        classifier_name = [classifier_name]
    for name in classifier_name:
        if hasattr(backbone_model, name):
            backbone_classifier_name = name
            break
    if backbone_classifier_name is None:
        raise AttributeError(f"No classifier found with names {classifier_name}")

    # Dynamically create the new class to inherit from the backbone model
    class ExtendableModel(backbone_model.__class__):
        """
        This is a general and dynamically created class that inherits any PyTorch model and replaces the classifier with a HierarchicalClassifier.

        Args:
            path: The path to the state file to load from. If None, masks and class_handles must be specified.
            device: The device to place the module on
            dtype: The data type to use for the module
            masks: A list of masks to use for the hierarchical classification. Will be overwritten if path is specified.
            class_handles: A dictionary containing the class handles. Will be overwritten if path is specified.
            **kwargs: Additional keyword arguments to pass to the backbone model constructor
        """
        def __init__(self, path=None, device=None, dtype=None, masks=None, class_handles=None, **kwargs):
            # Instead of calling super().__init__(), we copy the backbone model's __dict__ to this class's __dict__,
            # this is done to avoid needing to either supply the initialization arguments to the backbone model or
            # somehow retrieve the arguments dynamically. It is not pretty though...
            self.__dict__ = deepcopy(backbone_model.__dict__)
            # Type checking
            if not isinstance(masks, list) and masks is not None:
                raise ValueError("masks must be a list or None")
            elif isinstance(masks, list):
                if not all([isinstance(mask, torch.Tensor) for mask in masks]):
                    raise ValueError("masks must be a list of torch.Tensors")
            if not isinstance(class_handles, dict) and class_handles is not None:
                raise ValueError("class_handles must be a dict or None")
            if path is not None and not isinstance(path, str):
                raise ValueError("path must be a string or None")
            elif isinstance(path, str):
                # Check if the path is valid
                if not os.path.exists(path):
                    raise ValueError("path does not exist")
            if device is None:
                raise ValueError("device must be specified")
            if dtype is None:
                raise ValueError("dtype must be specified")
            # Set the device and dtype of the model and move it to the device
            self.device = device
            self.dtype = dtype
            # Save the original forward and _get_name functions
            self._internal_classifier = [getattr(self, backbone_classifier_name)] # This is workaround to avoid the state_dict recognizing the _internal_classifier as a parameter

            # Load the state
            self._load(path=path, masks=masks, class_handles=class_handles)

            # Move the model to the device and dtype
            self = self.to(device=device, dtype=dtype)

        def _load(self, path : Union[str, None]=None, masks : Union[List[torch.Tensor], None]=None, class_handles : Union[dict, None]=None) -> None:
            """
            Function to load the state of the model from a zip file.

            Args:
                path: The path to the state file
                masks: A list of masks to use for the hierarchical classification. Will be overwritten if path is specified.
                class_handles: A dictionary containing the class handles. Will be overwritten if path is specified.
            
            Returns:
                None
            """
            # If path is specified, get the class_handles and masks from the file
            if path is not None:
                state = parse_state_file(path)
                masks = [mask.to(device=self.device, dtype=self.dtype) for mask in state["masks"]]
                class_handles = state["class_handles"]
            # If path is not specified, check that masks and class_handles are specified
            else:
                if masks is None:
                    raise ValueError("masks must be specified if path is not.")
                if class_handles is None:
                    raise ValueError("class_handles must be specified if path is not.")

            self.class_handles = deepcopy(class_handles)
            self.classifier_name = backbone_classifier_name
            self._replace_classifier(masks)
            self._path = path

            # Load the state if path is specified, there might not be a state_dict in state, so check for it
            if path is not None:
                if "state_dict" in state:
                    self.load_state_dict(state["state_dict"])
                else:
                    print("Initializing classifier with random weights.")

        def save_to_path(self, path : str) -> None:
            """
            Function to save the state of the model to a zip file.

            Args:
                path: The path to save the state to
                masks: A list of masks to use for the hierarchical classification
                class_handles: A dictionary containing the class handles

            Returns:
                None
            """
            with zipfile.ZipFile(path, "w") as zip_file:
                # Save the state_dict
                with zip_file.open("state_dict.pt", "w") as f:
                    torch.save(self.state_dict(), f)
                # Save the class_handles
                zip_file.writestr("class_handles.pkl", pickle.dumps(self.class_handles))
                # Save the masks
                with zip_file.open("masks.pt", "w") as f:
                    torch.save(self.classifier.masks, f)

        @property
        def classifier(self):
            return self._modules[self.classifier_name]

        @classifier.setter
        def classifier(self, value):
            setattr(self, self.classifier_name, value)

        def _replace_classifier(self, masks) -> None:
            # Logic to replace the classifier
            num_classes = self.class_handles["n_classes"]
            old_classifier = self.classifier
            if isinstance(old_classifier, nn.Sequential):
                num_features = [i for i in old_classifier.modules() if isinstance(i, nn.Linear)][0].in_features
            elif isinstance(old_classifier, nn.Linear):
                num_features = old_classifier.in_features
            else:
                raise ValueError(f"Classifier must be a nn.Sequential or nn.Linear, not {type(old_classifier)}")
            self.classifier = HierarchicalClassifier(
                num_features=num_features,
                num_classes=num_classes,
                masks=masks,
                device=self.device,
                dtype=self.dtype
            )

        def forward_flex(self, x, argmax_translate : Union[bool, None]=False) -> Union[dict, List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor], Tuple[dict, torch.Tensor]]:
            x = super().forward(x)
            if argmax_translate:
                if self.classifier.return_embeddings:
                    embeddings, x = x
                else:
                    embeddings = None
                argmax = [torch.argmax(xi, dim=1) for xi in x]
                conf   = [torch.max(xi, dim=1).values.exp() for xi in x]
                x = [[[self.class_handles["idx_to_class"][level][aj.item()], cj.item()] for aj, cj in zip(ai, ci)] for level, (ai, ci) in enumerate(zip(argmax, conf))]
                x = HierachicalPrediction(x)
                if embeddings is not None:
                    x = (x, embeddings)
            # INSERT CUSTOM LOGIC HERE
            return x
        
        def forward(self, x : torch.Tensor) -> List[torch.Tensor]:
            x = super().forward(x)
            # INSERT CUSTOM LOGIC HERE
            return x
        
        def _get_name(self):
            return super()._get_name() + "_Hierarchical"
        
        @property
        def return_embeddings(self):
            return self.classifier.return_embeddings
        
        @return_embeddings.setter
        def return_embeddings(self, value):
            self.classifier.return_embeddings = value

        def toggle_embeddings(self, value : Union[bool, None]=None):
            """
            Function to toggle returning embeddings from the model (flag : self.classifier.embeddings).

            Args:
                value: The value to overwrite the current value with. If None, the value is toggled.

            Returns:
                The original value of self.classifier.return_embeddings
            """
            orig_value = self.return_embeddings
            if value:
                assert isinstance(value, bool), ValueError("Value must be a boolean")
                self.return_embeddings = value
            else:
                self.return_embeddings = not self.return_embeddings
            return orig_value
        
        def __repr__(self) -> str:
            return self._get_name()
        
        def to(self, device=None, dtype=None):
            self.classifier = self.classifier.to(device=device, dtype=dtype)
            return super(ExtendableModel, self).to(device=device, dtype=dtype)
        
        def cuda(self, device=None):
            self.classifier = self.classifier.cuda(device=device)
            return super(ExtendableModel, self).cuda(device=device)

    # Return the new class constructor    
    return ExtendableModel
