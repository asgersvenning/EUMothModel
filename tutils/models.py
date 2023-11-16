import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
from copy import deepcopy

_UNSUPPORTED_MODELS = ['fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fcos_resnet50_fpn', 'keypointrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'mvit_v1_b', 'mvit_v2_s', 'raft_large', 'raft_small', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'swin3d_b', 'swin3d_s', 'swin3d_t', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32']

class HierarchicalClassifier(nn.Module):
    def __init__(self, num_features, num_classes, masks, class_handles=None, path=None, device=torch.device("cpu"), dtype=torch.bfloat16):
        super(HierarchicalClassifier, self).__init__()
        self.silu = nn.SiLU()
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(num_features, device=device, dtype=dtype)
        self.linear1 = nn.Linear(num_features, 1024, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(1024, device=device, dtype=dtype)
        self.dropout2 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(1024, 1024, device=device, dtype=dtype)
        self.bn3 = nn.BatchNorm1d(1024, device=device, dtype=dtype)
        self.dropout3 = nn.Dropout(0.1)
        self.linear3 = nn.Linear(1024, 512, device=device, dtype=dtype)
        self.bn4 = nn.BatchNorm1d(512, device=device, dtype=dtype)
        self.dropout4 = nn.Dropout(0.1)
        # self.linear_logits = [nn.Linear(512, ncls, device=device, dtype=dtype) for ncls in num_classes]
        self.leaf_logits = nn.Linear(512, num_classes[0], device=device, dtype=dtype)
        self.bn5 = nn.BatchNorm1d(num_classes[0], device=device, dtype=dtype)
        self.masks = masks.copy()
        self.class_handles = deepcopy(class_handles)
        self.return_embeddings = False
        
    def forward(self, x):
        if self.return_embeddings:
            embeddings = x.clone()
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.linear1(x)
        x = self.bn2(x)
        x = self.silu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.bn3(x)
        x = self.silu(x)
        x = self.dropout3(x)
        x = self.linear3(x)
        x = self.bn4(x)
        x = self.silu(x)
        x = self.dropout4(x)
        x = self.leaf_logits(x)
        x = self.silu(x)
        y = self.bn5(x)
        y0 = F.log_softmax(y, dim = 1)
        y1 = F.log_softmax(torch.logsumexp(y0.unsqueeze(2) + self.masks[0].T, dim = 1), dim = 1)
        y2 = F.log_softmax(torch.logsumexp(y1.unsqueeze(2) + self.masks[1].T, dim = 1), dim = 1)
        if self.return_embeddings:
            return [y0, y1, y2], embeddings
        else:
            return [y0, y1, y2]
        
    def toggle_embeddings(self, value=None):
        orig_value = self.return_embeddings
        if value:
            assert isinstance(value, bool), ValueError("Value must be a boolean")
            self.return_embeddings = value
        else:
            self.return_embeddings = not self.return_embeddings
        return orig_value
    
    def save_state(self, path):
        torch.save(self.state_dict(), path)
        pickle.dump(self.masks, open(path + ".masks", "wb"))
        pickle.dump(self.class_handles, open(path + ".class_handles", "wb"))

    def load_state(self, path):
        self.load_state_dict(torch.load(path))
        self.masks = pickle.load(open(path + ".masks", "rb"))
        self.class_handles = pickle.load(open(path + ".class_handles", "rb"))

def HierarchicalModel(model, model_args={}, path=None):
    """
    This function takes an arbitrary model from torchvision and replaces all but the backbone with a HierarchicalClassifier.

    Args:
        model: The name of the model to load
        path: The path to the saved HierarchicalClassifier state

    Returns:
        A HierarchicalClassifier model with the specified backbone
    """
    if not isinstance(model, str):
        if not isinstance(model, torch.nn.Module):
            raise ValueError("model must be a string or a torch.nn.Module")
        Warning("Using a custom model. This may not work as expected.")
    else:
        if model in _UNSUPPORTED_MODELS:
            raise ValueError("The model {} is not supported.".format(model))
        model = torchvision.models.get_model(model, **model_args)
    
    old_classifier = None
    try:
        old_classifier = model.get_submodule("classifier")
    except AttributeError:
        pass
    try:
        old_classifier = model.get_submodule("fc")
    except AttributeError:
        pass

    if old_classifier is None:
        raise AttributeError(f"{model._get_name()} should be supported, but does not have classifier!")
    
    
    
