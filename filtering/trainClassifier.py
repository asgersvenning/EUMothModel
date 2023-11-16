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
import sys, shutil, os, time
from tqdm import tqdm

import pickle

os.chdir("/home/ucloud/EUMothModel")
from tutils.class_handling import *
from pyremotedata.implicit_mount import *
from pyremotedata.dataloader import *

from copy import deepcopy

from tensorboardX import SummaryWriter

backend = None

def confusion_matrix(model, batch, step, writer):
    for level in range(3):
        batch[0] = batch[0].to(device=device, dtype=dtype)

        with torch.no_grad():
            pred = model(batch[0])
            pcls_probs = pred[level].softmax(-1).cpu().float()
            pcls = pcls_probs.argmax(-1).numpy()
            
            true = batch[1][level].cpu().numpy()

        cm = np.zeros((num_classes[level] + 1, num_classes[level] + 1))
        for p, t in zip(pcls, true):
            cm[p, t] += 1
        cm /= cm.sum(axis=0) + 1
        cm[cm == 0] = -np.inf
        # Which classes are present in the batch
        present_classes = np.unique(np.concatenate((pcls, true)))
        # Remove classes that are not present in the batch from the confusion matrix
        cm = cm[present_classes][:, present_classes]
        
        axis_length = cm.shape[0]
        figwidth = min(10, max(100, axis_length / 100))
        plt.figure(figsize=(figwidth, figwidth))
        plt.imshow(cm)
        plt.colorbar()
        plt.tight_layout()
        writer.add_figure(f'val/confmat/{["Species", "Genus", "Family"][level]}', plt.gcf(), global_step=step)
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.hist(pcls_probs.flatten().numpy(), bins=100)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.tight_layout()
        writer.add_figure(f'val/hist/{["Species", "Genus", "Family"][level]}', plt.gcf(), global_step=step)
        plt.close()

    return

# weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
class create_image_preprocessing:
    def __init__(self, weights):
        self.transform = weights.transforms.func(crop_size=256)
        self.isize = self.transform.resize_size[0]
        self.mean = self.transform.mean
        self.std = self.transform.std

    def __call__(self, images):
        """Preprocess images for EfficientNet."""
        images = torchvision.transforms.Resize((self.isize, self.isize), antialias=True)(images)
        return self.transform(images)

image_preprocessing = create_image_preprocessing(weights)

def denormalize(tensor, mean=image_preprocessing.mean, std=image_preprocessing.std):
    """Denormalize a tensor."""
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(torch.float32)
    std = torch.tensor(std).view(1, 3, 1, 1).to(torch.float32)
    return tensor.cpu().to(torch.float32) * std + mean

## Hyperparameters
skip = 0 # Number of batches to skip (used to resume script from a specific batch)
batch_size = 128 # Batch size for chunked loading of images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype=torch.bfloat16
real_class_index = 0 # The real class index of the species to fine-tune on (6 for the initial model; insectGBIF-1280m6.pt)
inference_size = 640 # The size of the images to run inference on (1280 for the initial model; insectGBIF-1280m6.pt)

data_split = 95., 1., 4.

# ERDA data transfer setup
if backend is not None:
    backend.stop()
backend = IOHandler(verbose = False, clean=True)
backend.start()
backend.cd("AMI_GBIF_Pretraining_Data/rebalanced75_without_larvae")
backend.cache_file_index(skip=skip)

# Dataset and dataloader setup
remote_iterator = RemotePathIterator(
        backend,
        batch_size=128,
        max_queued_batches=5,
        n_local_files=5*128*2,
    )

full_dataset = RemotePathDataset(
    remote_iterator,
    prefetch=1*batch_size,
    transform=image_preprocessing,
    # device=device, 
    dtype=dtype, 
    hierarchical=True,
    return_remote_path=False,
    return_local_path=True,
    verbose=False
)

class_counts = class_counting(full_dataset.remote_path_iterator.remote_paths, full_dataset.class_handles)
hierarchy = create_hierarchy(full_dataset.remote_path_iterator.remote_paths, full_dataset.class_handles)
masks = mask_hierarchy(hierarchy, device=device, dtype=dtype)

children = [mask.exp().sum(dim=1) for mask in masks]
children = [torch.ones((full_dataset.n_classes[0]))] + children
children = [i / i.mean() for i in children]
children = [i.to(device=device, dtype=dtype) for i in children]



ri_train, ri_val, ri_test = remote_iterator.split(proportion=[0.95, 0.01, 1-0.95-0.01])
datasets = [RemotePathDataset(
    ri,
    prefetch=3*batch_size,
    transform=image_preprocessing,
    # device=device, 
    dtype=dtype,
    hierarchical=True,
    return_remote_path=False,
    return_local_path=True,
    verbose=False
) for ri in [ri_train, ri_val, ri_test]]

datasets[0].class_handles = full_dataset.class_handles
datasets[1].class_handles = full_dataset.class_handles
datasets[2].class_handles = full_dataset.class_handles
train, val, test = [CustomDataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8) for dataset in datasets]

lengths = [len(dataset) for dataset in datasets]

# Model definition
model = torchvision.models.efficientnet_b0(weights=None).half().train(False)
num_features = [k for k in [j for j in [i for i in model.children()][0].children()][-1].children()][0].out_channels
num_classes = datasets[0].n_classes

class HierarchicalClassifier(nn.Module):
    def __init__(self, num_features, num_classes, masks, class_handles=None):
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



model.to(device=device, dtype=dtype)
model.classifier = HierarchicalClassifier(num_features, num_classes, masks)
model.eval()

model_checkpoint = torch.load("/home/ucloud/EUMothModel/models/run19/epoch_4_batch_final.pt", map_location=device)
model.load_state_dict(model_checkpoint)

epochs = 1
lr = 1 * 1e-3

# loss_fn = [nn.CrossEntropyLoss(weight=1/children[i], reduction="none") for i in range(3)]

optimizer = optim.AdamW([
    {
        'params' : model.classifier.parameters(),
        'lr' : lr,
        'weight_decay' : 1e-5,
    },
    {
        'params' : model.features.parameters(),
        'lr' : lr,
        'weight_decay' : 1e-5,
    }])

# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(1e-3)**(1/(epochs*len(train))), verbose=False)
# lr_scheduler1 = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train), pct_start=0.01, div_factor=1e3, final_div_factor=1e0, verbose=False)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=False)

family_weight, genus_weight, species_weight = 1, 3, 10
loss_weights = [species_weight, genus_weight, family_weight]
loss_weights = [len(loss_weights) * i / sum(loss_weights) for i in loss_weights]

class_weights = [torch.tensor(i, device=device, dtype=dtype) for i in class_counts]
class_weights = [1 / (i + 1).log10() for i in class_weights]
class_weights = [i / i.mean() for i in class_weights]
for i in class_weights:
    i.requires_grad = False
loss_fn = [nn.CrossEntropyLoss(weight=1/children[i], reduction="none") for i in range(3)]

[print('{} : min={:.2f} | max={:.2f} | mean={:.2f} | std={:.2f}'.format(["species", "genus  ", "family "][ind], i.min().item(), i.max().item(), i.mean().item(), i.std().item())) for ind, i in enumerate(class_weights)]


# tensorboard --logdir=/home/ucloud/EUMothModel/tensorboard_logs
# Define the base log directory
base_logdir = "/home/ucloud/EUMothModel/tensorboard_logs"

# List all existing run directories in the base log directory
existing_runs = [d for d in os.listdir(base_logdir) if d.startswith("run")]

# Find the maximum run number among existing directories and increment for the next run
next_run_number = max([int(run[3:]) for run in existing_runs] + [0]) + 1
run_name = f"run{next_run_number}"
# run_name = "best_lr"

if not os.path.exists(f"/home/ucloud/EUMothModel/models/{run_name}") and run_name != "best_lr":
    os.makedirs(f"/home/ucloud/EUMothModel/models/{run_name}")

# Create a new run directory using the incremented run number
# logdir = os.path.join(base_logdir, f"run{next_run_number}")
logdir = os.path.join(base_logdir, run_name)
if run_name == "best_lr":
    # Delete the existing run directory if we are using the best_lr run
    if os.path.exists(logdir):
        print(f"Deleting existing logdir {logdir}")
        shutil.rmtree(logdir)
        os.makedirs(logdir)

# Create the SummaryWriter with the unique logdir
writer = SummaryWriter(logdir=logdir, flush_secs=10)
writer.flush()

break_flag = False

for epoch in range(0, epochs):
    model.train()

    torch.cuda.empty_cache()
    
    pbar = tqdm(train, total=len(train), desc=f"Epoch {epoch}")
    for batch_i, batch in enumerate(pbar):
        images, labels, paths = batch
        images = images.to(device=device, dtype=dtype)

        wcls = [class_weights[i][labels[i]] for i in range(3)]
        wcls = [i / i.mean() for i in wcls]
        
        optimizer.zero_grad()
        pred = model(images)
        del images
        loss = [(loss_fn[i](pred[i], labels[i].to(device)) * wcls[i]).mean()  for i in range(3)]
        del pred
        combined_loss = sum([loss[i] * loss_weights[i] for i in range(3)])
        combined_loss.backward()
        
        optimizer.step()
        lr_scheduler.step(combined_loss)

        this_lr = optimizer.param_groups[0]['lr']
        
        for i, ltype in enumerate(["species", "genera", "families"]):
            writer.add_scalar(f"Loss/train/{ltype}", loss[i].cpu().float(), pbar.n + len(train) * epoch)
        writer.add_scalar("Loss/train/combined", combined_loss.cpu().float(), pbar.n + len(train) * epoch)
        writer.add_scalar("LR", this_lr, pbar.n + len(train) * epoch)
        
        pbar.set_description_str(f"TRAIN: Epoch {epoch} | Family : {loss[2]:.2f} | Genus : {loss[1]:.2f} | Species : {loss[0]:.2f} | Combined : {combined_loss:.2f} | LR: {this_lr:.3E} | Memory usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        del loss, combined_loss
        if batch_i % 2000 == 0 and batch_i != 0:
            torch.save(model.state_dict(), f"/home/ucloud/EUMothModel/models/{run_name}/epoch_{epoch}_batch_{batch_i}.pt")
        if batch_i == 100 and run_name == "best_lr":
            break_flag = True
        writer.flush()
        if break_flag:
            break

    # Save checkpoint after each epoch
    if run_name != "best_lr":
        torch.save(model.state_dict(), f"/home/ucloud/EUMothModel/models/{run_name}/epoch_{epoch}_batch_final.pt")

    time.sleep(1)

    # Evaluate on validation set after each epoch and log metrics to TensorBoard (done after saving checkpoint, to ensure that errors in validation do not prevent checkpoint from being saved)
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        pbar = tqdm(val, total=len(val), desc=f"Validation {epoch}")
        for batch_i, batch in enumerate(pbar):
            images, labels, paths = batch
            images = images.to(device=device, dtype=dtype)

            wcls = [class_weights[i][labels[i]] for i in range(3)]
            wcls = [i / i.mean() for i in wcls]

            pred = model(images)
            del images
            loss = [(loss_fn[i](pred[i], labels[i].to(device)) * wcls[i]).mean() for i in range(3)]
            del pred
            combined_loss = sum([loss[i] * loss_weights[i] for i in range(3)])

            for i, ltype in enumerate(["species", "genera", "families"]):
                writer.add_scalar(f"Loss/val/{ltype}", loss[i].cpu().float(), pbar.n + len(train) * epoch)
            writer.add_scalar("Loss/val/combined", combined_loss.cpu().float(), pbar.n + len(train) * epoch)

            pbar.set_description_str(f"VAL: Epoch {epoch} | Family : {loss[2]:.2f} | Genus : {loss[1]:.2f} | Species : {loss[0]:.2f} | Combined : {combined_loss:.2f} | Memory usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            del loss, combined_loss
            torch.cuda.empty_cache()
            if batch_i == 0:
                confusion_matrix(model, batch, epoch, writer)
            writer.flush()
            if break_flag:
                break
    
    if break_flag:
        break
    
    time.sleep(1)
model.eval()


max_width = 30
def format_text(text):
    if len(text) > max_width:
        # Truncate long text
        return text[:(max_width-3)] + "..."
    else:
        # Pad and center shorter text
        padding = max_width - len(text)
        left_padding = padding // 2
        right_padding = padding - left_padding
        return " " * left_padding + text + " " * right_padding

def PlotPredictions(model, loader, n):
    if model.training:
        model_is_training = True
        model.eval()
    else:
        model_is_training = False

    cls_dicts = train.dataset.idx_to_class
    fam_dict, gen_dict, sp_dict = cls_dicts[2], cls_dicts[1], cls_dicts[0]

    with torch.no_grad():
        torch.cuda.empty_cache()
        for batch in loader:
            images, labels, paths = batch
            images = images.to(device=device, dtype=dtype)

            pred = model(images)
            for i in range(len(images)):
                pfamily, pgenus, pspecies = pred[2][i].argmax(), pred[1][i].argmax(), pred[0][i].argmax()
                tfamily, tgenus, tspecies = labels[2][i], labels[1][i], labels[0][i]

                pfamily, tfamily, pgenus, tgenus, pspecies, tspecies = pfamily.cpu().item(), tfamily.cpu().item(), pgenus.cpu().item(), tgenus.cpu().item(), pspecies.cpu().item(), tspecies.cpu().item()
                print(
                    "pfamily:", pfamily, "tfamily:", tfamily, "pgenus:", pgenus, "tgenus:", tgenus, "pspecies:", pspecies, "tspecies:", tspecies, "\n"
                )
                pfamily, tfamily, pgenus, tgenus, pspecies, tspecies = fam_dict[pfamily], fam_dict[tfamily], gen_dict[pgenus], gen_dict[tgenus], sp_dict[pspecies], sp_dict[tspecies]

                pred_v_truth = f'''
                {"":<9} | {format_text("Predicted"):<{max_width}} | {format_text("Truth"):<{max_width}}
                {"-" * 9} | {"-" * max_width} | {"-" * max_width}
                {"Family":^10}| {format_text(pfamily):<{max_width}} | {format_text(tfamily):<{max_width}}
                {"Genus":^10}| {format_text(pgenus):<{max_width}} | {format_text(tgenus):<{max_width}}
                {"Species":^10}| {format_text(pspecies):<{max_width}} | {format_text(tspecies):<{max_width}}
                '''

                img = images[i].cpu().float()
                img = denormalize(img).squeeze()

                img = img.clamp(0, 1)
                img = img.permute(1, 2, 0)

                # Create a figure with a white background
                fig = plt.figure(figsize=(6, 6), facecolor='white')

                # Create an Axes to hold your image
                ax = fig.add_axes([0, 0, 1, 1])

                # Plot your image
                ax.imshow(img)

                # Print predictions vs ground truth as the plot title, centered and in monospace
                plt.title(pred_v_truth, fontdict={'fontsize': 8, 'fontfamily': 'monospace'}, loc='center')
                plt.axis('off')

                # Save the plot with a white background
                plt.savefig(f'/home/ucloud/EUMothModel/plots/prediction_{n}.png', facecolor='white', dpi=300, bbox_inches='tight', pad_inches=0.25)

                # Close the figure
                plt.close(fig)

                n -= 1
                if n == 0:
                    break
            if n == 0:
                break
    if model_is_training:
        model.train()
    else:
        model.eval()

    return images, labels, pred

test = PlotPredictions(model, val, 8)

timg, tlabels, tpred = test
timg = timg[0]
tlabels = [i[0].item() for i in tlabels]
tpred = [i[0] for i in tpred]
tpred_label = [i.argmax().item() for i in tpred]
tpred_readable = [train.dataset.idx_to_class[i][j] for i, j in enumerate(tpred_label)]
treadable_labels = [train.dataset.idx_to_class[i][j] for i, j in enumerate(tlabels)]


tmpred = [tpred[0].softmax(-1)]
for i, mask in enumerate(masks):
    tmpred.append(tmpred[-1] @ mask.T)
tmpred_label = [i.argmax().item() for i in tmpred]
tmpred_readable = [train.dataset.idx_to_class[i][j] for i, j in enumerate(tmpred_label)]

print("True:", treadable_labels)
print("Prediction:", tpred_readable)
print("Forced Correct Prediction:", tmpred_readable)

backend.stop()