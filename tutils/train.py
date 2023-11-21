from .models import create_extendable_model_class
from .plots import confusion_matrix
from .class_handling import *

from pyremotedata.implicit_mount import *
from pyremotedata.dataloader import *

from tensorboardX import SummaryWriter

import torch, torchvision
import torch.optim as optim
import torch.nn as nn

import argparse

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="bfloat16")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--path", type=str, default="models/empty.state")
parser.add_argument("--model", type=str, default="efficientnet_b0")
parser.add_argument("--skip", type=int, default=0)
parser.add_argument("--name", type=str, default="")

if __name__ == "__main__":
    args = parser.parse_args()

    # Device and dtype
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # Hyperparameters
    batch_size = args.batch_size
    model_type = args.model
    num_workers = args.n_workers
    epochs = args.n_epochs
    lr = args.lr

    # Run name and logdir
    base_logdir = "tensorboard_logs"
    existing_runs = [d for d in os.listdir(base_logdir) if d.startswith("run")]
    run_name = args.name
    if run_name == "":
        # Find the maximum run number among existing directories and increment for the next run
        next_run_number = max([int(run[3:]) for run in existing_runs] + [0]) + 1
        run_name = f"run{next_run_number}"

    if not os.path.exists(f"models/{run_name}") and run_name != "best_lr":
        os.makedirs(f"models/{run_name}")

    # Create a new run directory using the incremented run number
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

    # Model definition
    backbone = model_type
    constructor = create_extendable_model_class(backbone_model=backbone)
    model = constructor(path=args.path, device=device, dtype=dtype)

    # Preprocessor
    preprocessor = torchvision.models.get_model_weights(model_type).transforms

    # Dataloaders
    backend = IOHandler(verbose = False, clean=True)
    backend.start()
    backend.cd("AMI_GBIF_Pretraining_Data/rebalanced75_without_larvae")
    backend.cache_file_index(skip=args.skip)

    # Dataset and dataloader setup
    remote_iterator = RemotePathIterator(
            backend,
            batch_size=128,
            max_queued_batches=5,
            n_local_files=5*128*2,
        )

    full_dataset = RemotePathDataset(
        remote_iterator,
        prefetch=batch_size,
        transform=preprocessor,
        device=torch.device("cpu"),
        dtype=dtype, 
        hierarchical=True,
        return_remote_path=False,
        return_local_path=True,
        verbose=False
    )

    ri_train, ri_val, ri_test = remote_iterator.split(proportion=[0.95, 0.01, 1-0.95-0.01])
    datasets = [RemotePathDataset(
        ri,
        prefetch=3*batch_size,
        transform=preprocessor,
        device=torch.device("cpu"),
        dtype=dtype,
        hierarchical=True,
        return_remote_path=False,
        return_local_path=True,
        verbose=False
    ) for ri in [ri_train, ri_val, ri_test]]

    datasets[0].class_handles = full_dataset.class_handles
    datasets[1].class_handles = full_dataset.class_handles
    datasets[2].class_handles = full_dataset.class_handles
    train, val, test = [CustomDataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) for dataset in datasets]

    lengths = [len(dataset) for dataset in datasets]

    # Loss function and optimizer    
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

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=False)

    family_weight, genus_weight, species_weight = 1, 3, 10
    loss_weights = [species_weight, genus_weight, family_weight]
    loss_weights = [len(loss_weights) * i / sum(loss_weights) for i in loss_weights]

    class_counts = class_counting(full_dataset.remote_path_iterator.remote_paths, model.class_handles)
    class_weights = [torch.tensor(i, device=device, dtype=dtype) for i in class_counts]
    class_weights = [1 / (i + 1).log10() for i in class_weights]
    class_weights = [i / i.mean() for i in class_weights]
    for i in class_weights:
        i.requires_grad = False
    loss_fn = [nn.CrossEntropyLoss(weight=1/children[i], reduction="none") for i in range(3)]

    print("Loss weight summary:")
    [print('{} : min={:.2f} | max={:.2f} | mean={:.2f} | std={:.2f}'.format(["species", "genus  ", "family "][ind], i.min().item(), i.max().item(), i.mean().item(), i.std().item())) for ind, i in enumerate(class_weights)]



    # Flag to break out of training loop early
    break_flag = False

    # Training hyperparameters
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