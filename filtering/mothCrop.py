import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from math import sqrt, ceil
import sys
from tqdm import tqdm

# Import YOLOv5 helper functions
sys.path.append('/home/ucloud/EUMothModel')
from tutils.yolo_helpers import image_preprocessing, box_iou, xywh2xyxy, non_max_suppression
from tutils.implicit_mount import *
from tutils.dataloader import *

model_weights = "insect_iter7-1280m6.pt"
model = torch.hub.load(
    'ultralytics/yolov5', 
    'custom', 
    path=f'models/{model_weights}',
    force_reload=False)

## Hyperparameters
skip = 0 # Number of batches to skip (used to resume script from a specific batch)
batch_size = 8 # Batch size for chunked loading of images
n_batches = 8 # Number of batches to load
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype=torch.float16
real_class_index = 0 # The real class index of the species to fine-tune on (6 for the initial model; insectGBIF-1280m6.pt)
inference_size = 640 # The size of the images to run inference on (1280 for the initial model; insectGBIF-1280m6.pt)

# ERDA data transfer setup
backend = IOHandler(verbose = False, clean=True)
backend.start()
backend.cd("AMI_GBIF_Pretraining_Data/rebalanced75_without_larvae")
backend.cache_file_index(skip=skip)

# Dataset and dataloader setup
dataset = RemotePathDataset(
    remote_path_iterator=RemotePathIterator(
        backend,
        batch_size=128,
        max_queued_batches=5,
        n_local_files=5*128*2,
    ),
    prefetch=5*batch_size,
    transform=image_preprocessing,
    device=device,
    dtype=dtype,
    return_remote_path=False,
    return_local_path=True,
    verbose=False
)
dataloader = CustomDataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

length = len(dataset)

model = model.to(device,dtype=dtype)

# Test the model
model.eval()

# Inference
predictions = []
images = []

with torch.no_grad():
    for i, (inputs, labels, paths) in tqdm(enumerate(dataloader), total=n_batches):
        if i >= n_batches:
            break
        # Forward pass
        pred = model(inputs / 255, augment=True)
        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = [i for i in pred if i is not None]
            pred = torch.cat(pred, dim=1) # TODO: Why does this even work?
        pred = pred.clone().to(torch.float32)
        pred[..., 4][torch.argmax(pred[..., 5:], dim=-1) != real_class_index] = -1
        # Postprocess outputs
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=5)

        # Append predictions
        predictions += pred

        # Insert images
        images += paths


## Plot predictions for all images
# Make plot directory if it doesn't exist
if not os.path.isdir("plots"):
    os.mkdir("plots")

print(len(images), len(predictions))
assert len(images) == len(predictions), "Number of images and predictions must be equal"

# # Plot predictions
# for i, (image, prediction) in enumerate(zip(images, predictions)):
#     image = read_image(image)
    
#     # Plot the image
#     plt.imshow(image.cpu().permute(1, 2, 0))
    
#     height, width = image.shape[1:]
#     confidences = prediction[..., 4]
#     max_confidence = torch.max(confidences)
    
#     for box in prediction:
#         x1, y1, x2, y2, conf, cls = box
#         x1, y1, x2, y2 = x1 * width/inference_size, y1 * height/inference_size, x2 * width/inference_size, y2 * height/inference_size
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
#         # Set the color and alpha based on confidence
#         alpha = conf.item() / 3 + 0.025  # Use confidence for alpha
#         color = 'tab:red' if conf.item() == max_confidence else 'tab:gray'
        
#         # Plot the bounding box as a semi-transparent rectangle
#         rectangle = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True, color=color, alpha=alpha)
#         plt.gca().add_patch(rectangle)
    
#     plt.axis('off')
#     plt.savefig(f"plots/{i}.png", transparent=True)
#     plt.close()


# Define the number of rows and columns for the grid
num_rows = int(sqrt(len(images)))
num_cols = ceil(len(images) / num_rows)

# Calculate the total number of subplots
total_plots = len(images)

# Create a figure with a grid of subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 8, num_rows * 8))

# Iterate through the images and predictions
for i, (image, prediction) in enumerate(zip(images, predictions)):
    row = i // num_cols
    col = i % num_cols
    
    image = read_image(image)
    
    # Set the current subplot
    ax = axs[row, col]
    
    # Plot the image
    ax.imshow(image.cpu().permute(1, 2, 0))
    
    height, width = image.shape[1:]
    confidences = prediction[..., 4]
    max_confidence = torch.max(confidences)
    
    for box in prediction:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = x1 * width/inference_size, y1 * height/inference_size, x2 * width/inference_size, y2 * height/inference_size
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Set the color and alpha based on confidence
        alpha = conf.item() / 3 + 0.025  # Use confidence for alpha
        color = 'tab:red' if conf.item() == max_confidence else 'tab:gray'
        
        # Plot the bounding box as a semi-transparent rectangle
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True, color=color, alpha=alpha))
    
    ax.axis('off')

# Remove any empty subplots
for i in range(total_plots, num_rows * num_cols):
    ax = axs[i // num_cols, i % num_cols]
    ax.axis('off')

# Adjust the layout and save the single large plot
plt.tight_layout()
plt.savefig("plots/combined_plot.png", transparent=True)