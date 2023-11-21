import torch
import torchvision
import torchdynamo

from torch.utils.benchmark import Timer

from math import sqrt, ceil
import os, sys
import time

import sahi.predict as predict

# Import YOLOv5 helper functions
os.chdir("/home/ucloud/EUMothModel")
sys.path.append("/home/ucloud/EUMothModel")
from tutils.models import *

dtype = torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
backbone = "efficientnet_b0"
constructor = create_extendable_model_class(backbone_model=backbone)
model_normal = constructor(path="models/run19.state", device=device, dtype=dtype)
# model_compile = torch.compile(model_normal)
model_compile = deepcopy(model_normal)
model_compile = torch.compile(model_compile, fullgraph=False, backend="inductor", 
                              mode="default",
                            #   options={
                            #       "triton.cudagraphs" : True,
                            #       "shape_padding" : True,
                            #   }
                              ) # ['aot_ts_nvfuser', 'cudagraphs', 'inductor', 'ipex', 'nvprims_nvfuser', 'onnxrt', 'tvm']

def generate_test_image(n, dim=256, device=device, dtype=dtype):
    """
    Generates a test image of size (n, 1, dim, dim) with values in [0, 1]
    """
    return torch.rand((n, 3, dim, dim), device=device, dtype=dtype)

inputs = generate_test_image(32, device=device, dtype=dtype)

model_normal(inputs)
model_normal.zero_grad()

model_compile(inputs)
model_compile.zero_grad()

# Warmup models
with torch.no_grad():
    model_normal(inputs)
    model_compile(inputs)

# Timed model execution function
def time_model(model, inputs, warmup=5, n=100):
    """
    Times the execution of a model on a given input
    """
    model.eval()
    with torch.no_grad():
        for i in range(n + warmup):
            if i == warmup:
                start = time.time() 
            model(inputs)
        end = time.time()
    return (end - start) / n

# Without gradients
with torch.no_grad():
    normal_result = time_model(model_normal, inputs)
    compile_result = time_model(model_compile, inputs)

print("Without gradients:")
print(f"Normal: {normal_result:.3f}s | Compile: {compile_result:.3f}s")

def time_model_forward_backward(model, inputs, warmup=5, n=10):
    """
    Times the execution of a model on a given input
    """
    model.train()
    for i in range(n + warmup):
        if i == warmup:
            start = time.time()
        model.zero_grad()
        outputs = model(inputs)
        loss = outputs[0].sum()
        loss.backward()
    end = time.time()
    return (end - start) / n

# With gradients
normal_result = time_model_forward_backward(model_normal, inputs)
compile_result = time_model_forward_backward(model_compile, inputs)

print("With gradients:")
print(f"Normal: {normal_result:.3f}s | Compile: {compile_result:.3f}s")