# Import standard modules
import argparse
import os, sys
import math
from copy import deepcopy

# Import PyTorch modules
import torch, torchvision
import torch.optim as optim
import torch.nn as nn

# Import monitoring modules
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Import custom modules
from pyremotedata.implicit_mount import *
from pyremotedata.dataloader import *

# Import utility modules
sys.path.append(os.path.dirname(__file__))
# sys.path.append("/home/ucloud/EUMothModel/tutils")
from models import create_extendable_model_class
from plots import confusion_matrix
from class_handling import *

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="bfloat16")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--warmup", type=int, default=100)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--path", type=str, default="models/empty.state")
parser.add_argument("--model", type=str, default="efficientnet_b0")
parser.add_argument("--skip", type=int, default=0)
parser.add_argument("--name", type=str, default="")
parser.add_argument("--schedule", type=str, default="one_cycle")
parser.add_argument("--subsample", type=int, default=1)
parser.add_argument("--dataset", type=str, default="rebalanced75_without_larvae")
parser.add_argument("--tunelr", action="store_true")
parser.add_argument("--val_only", action="store_true")
                    
if __name__ == "__main__":
    args = parser.parse_args()

    # Device and dtype
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    if args.val_only and args.tunelr:
        raise ValueError("Cannot use both --val_only and --tunelr")

    # Hyperparameters
    batch_size = args.batch_size
    model_type = args.model
    num_workers = args.n_workers
    epochs = args.n_epochs
    warmup = args.warmup
    lr = args.lr
    family_weight, genus_weight, species_weight = 1, 3, 10

    if args.tunelr:
        epochs = 1
        lr = 10
    elif args.val_only:
        epochs = 1

    # Run name and logdir
    base_logdir = "tensorboard_logs"
    existing_runs = [d for d in os.listdir(base_logdir) if d.startswith("run")]
    run_name = args.name
    if run_name == "":
        # Find the maximum run number among existing directories and increment for the next run
        next_run_number = max([int(run[3:]) for run in existing_runs if run.startswith("run")] + [0]) + 1
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
    constructor = create_extendable_model_class(
        backbone_model=backbone, 
        model_args={"weights": "IMAGENET1K_V1"}
    )
    model = constructor(path=args.path, device=device, dtype=dtype)
    
    # If learning rate is not specified, guess a good one
    if lr == 0:
        print("No learning rate specified, guessing...")
        lr = round(0.002 * 5 / (4 + model.class_handles["n_classes"][0]), 6)
        print(f"Learning rate guess: {lr}")

    # Preprocessor
    model_transform = torchvision.models.get_model_weights(model_type).DEFAULT.transforms(antialias=True)
    preprocessor = lambda x : model_transform(x / 255.0)

    if args.dataset == "rebalanced75_without_larvae":
        # Dataloaders
        backend = IOHandler(verbose = False, clean=True)
        backend.start()
        backend.cd("rebalanced75_without_larvae")
        backend.cache_file_index(skip=args.skip)

        # Dataset and dataloader setup
        remote_iterator = RemotePathIterator(
                backend,
                batch_size=128,
                max_queued_batches=5,
                n_local_files=5*128*2,
            )

        # Subset and shuffle the dataset (this is non-deterministic at the moment, adding support for a seed is a TODO)
        remote_iterator.shuffle()
        if args.subsample < 1:
            raise ValueError("Subsampling less than 1 is non-sensical.")
        elif args.subsample != 1:
            subset_indices = np.random.choice(len(remote_iterator), size=int(len(remote_iterator) / args.subsample), replace=False).tolist()
            remote_iterator.subset(subset_indices)

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
        # Set the class handles for the dataset
        full_dataset.class_handles = deepcopy(model.class_handles)

        ri_train, ri_val, ri_test = remote_iterator.split(proportion=[0.95, 0.01, 1-0.9-0.01])
        datasets = [RemotePathDataset(
            ri,
            prefetch=256,
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
    elif args.dataset == "MNIST":
        def mnist_to_hierarchy(x):
            if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
                x = x.tolist()
            if not isinstance(x, list):
                x = [x]
            cls_leaf = [model.class_handles["idx_to_class"][0][i] for i in x]
            cls_names = [None] * len(cls_leaf)
            for idx, i in enumerate(cls_leaf):
                for j in model.class_handles["classes"][1]:
                    if not i in j:
                        continue
                    for k in model.class_handles["classes"][2]:
                        if not i in k:
                            continue
                        cls_names[idx] = i, j, k
            cls_idx = [model.class_handles["class_to_idx"][level][name] for tcls in cls_names for level, name in enumerate(tcls)]
            return cls_idx
        
        def mnist_class_counts(targets, target_transform):
            from collections import Counter
            counts = [Counter(range(i)) for i in model.class_handles["n_classes"]]
            for target in targets:
                for ctype, class_idx in enumerate(target_transform(target)):
                    counts[ctype][class_idx] += 1
            counts = [np.array(list(counter.values())) for counter in counts]
            return counts
        
        class MNIST_wrapper(torchvision.datasets.MNIST):
            def __getitem__(self, index):
                img, target = super().__getitem__(index)
                return img, target, [1]
        
        toT = torchvision.transforms.PILToTensor()
        mnist_preprocessor = lambda x : preprocessor(toT(x).repeat(3, 1, 1).float())
        
        # Dataset and dataloader setup
        full_dataset = train = MNIST_wrapper(
            root="data", 
            train=True, 
            download=True, 
            transform=mnist_preprocessor,
            target_transform=mnist_to_hierarchy
        )
        full_dataset.class_handles = deepcopy(model.class_handles)

        datasets = torch.utils.data.random_split(full_dataset, [int(0.1 * len(full_dataset)), int(0.01 * len(full_dataset)), len(full_dataset) - int(0.1 * len(full_dataset)) - int(0.01 * len(full_dataset))])
        dtrain, dval, dtest = datasets
        train = torch.utils.data.DataLoader(dtrain, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val = torch.utils.data.DataLoader(dval, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")


    lengths = {t : len(dataset) for t, dataset in zip(["train", "val", "test"], datasets)}

    # Loss function and optimizer    
    optimizer = optim.AdamW(tuple(model.parameters()), lr=1, betas=(args.momentum, 0.999), weight_decay=1e-5)

    lambda_fn = None
    # Inspired by: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L364
    def one_cycle(max_lr, min_lr, steps, warmup_steps, eps=None, warmup_fn=lambda x : x ** 2):
        if eps is None:
            eps  = min_lr / warmup_steps * 1e-4
        def _one_cycle(step):
            # Instead of using a step function, we use exponential interpolation
            # to smoothly transition from warmup to cosine
            warmup_weight = math.exp(-2 * (0 if step < warmup_steps else (2 * step) / warmup_steps)) 
            cosine_weight = 1 - warmup_weight
            
            cosine_lambda = max_lr * (1 + math.cos(math.pi * (step - warmup_steps)/(steps - warmup_steps))) / 2 + min_lr

            # Only calculate warmup when warmup_weight is significant
            if warmup_weight > 0.5:
                warmup_lambda = max_lr * warmup_fn(step / warmup_steps)/warmup_fn(1) + eps
                return warmup_weight * warmup_lambda + cosine_weight * cosine_lambda
            else:
                return cosine_lambda
        
        return _one_cycle

    if args.tunelr:
        warmup = len(train) - 1

    if args.schedule == "one_cycle":
        print("Using one-cycle learning rate schedule")
        lambda_fn = one_cycle(lr, lr / 1e3, epochs * len(train), warmup, None, lambda x : 2 ** (x * 30))
    elif args.schedule == "linear":
        print("Using linear learning rate schedule")
        lambda_fn = lambda step: 1 - step  / (epochs * len(train))
    else:
        raise NotImplementedError(f"Learning rate schedule {args.schedule} not implemented")

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)

    loss_weights = [species_weight, genus_weight, family_weight]
    loss_weights = [len(loss_weights) * i / sum(loss_weights) for i in loss_weights]

    if args.dataset == "rebalanced75_without_larvae":
        class_counts = class_counting(ri_train.remote_paths, full_dataset.class_handles)
    elif args.dataset == "MNIST":
        class_counts = mnist_class_counts(full_dataset.targets[dtrain.indices], mnist_to_hierarchy)
    class_weights = [torch.tensor(i, device=device, dtype=dtype) for i in class_counts]
    class_weights = [1 / (i + i.mean()) for i in class_weights]
    class_weights = [i / i.mean() for i in class_weights]
    # class_weights = [torch.ones_like(i) for i in class_weights] # Uncomment this line to disable class weighting
    for i in class_weights:
        i.requires_grad = False
    loss_fn = [
        nn.CrossEntropyLoss(
            weight=class_weights[i], 
            reduction="none", 
            # label_smoothing=1/class_weights[i].shape[0]
        ) for i in range(3)
    ]

    print("Loss weight summary:")
    [print('{} : min={:.2f} | max={:.2f} | mean={:.2f} | std={:.2f}'.format(["species", "genus  ", "family "][ind], i.min().item(), i.max().item(), i.mean().item(), i.std().item())) for ind, i in enumerate(class_weights)]

    # Flag to break out of training loop early
    break_flag = False

    # Training hyperparameters
    for epoch in range(0, epochs):
        model.train()

        torch.cuda.empty_cache()
        
        if not args.val_only:
            pbar = tqdm(train, total=len(train), desc=f"Epoch {epoch}", dynamic_ncols=True)
        else:
            pbar = []
        for batch_i, batch in enumerate(pbar):
            images, labels, paths = batch # TODO: Remove paths from batch (need to change CustomDataLoader to not return paths, or perhaps make the batch a dict with keys "images" and "labels" as well as any other keys that may be needed in other scenarios)
            images = images.to(device=device, dtype=dtype)

            # Skip batch if it has 1 or 0 images
            if images.shape[0] <= 1:
                continue

            wcls = [class_weights[i][labels[i]] for i in range(3)]
            # wcls = [i / i.mean() for i in wcls]
            
            optimizer.zero_grad()
            pred = model(images)
            del images
            loss = [(loss_fn[i](pred[i], labels[i].to(device)) * wcls[i]).mean()  for i in range(3)]
            del pred
            combined_loss = sum([loss[i] * loss_weights[i] for i in range(3)])
            combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
            
            optimizer.step()
            this_lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()
            
            for i, ltype in enumerate(["species", "genera", "families"]):
                writer.add_scalar(f"Loss/train/{ltype}", loss[i].cpu().float(), pbar.n + len(train) * epoch)
            writer.add_scalar("Loss/train/combined", combined_loss.cpu().float(), pbar.n + len(train) * epoch)
            writer.add_scalar("LR", this_lr, pbar.n + len(train) * epoch)
            
            pbar.set_description_str(f"TRAIN: Epoch {epoch} | Family : {loss[2]:.2f} | Genus : {loss[1]:.2f} | Species : {loss[0]:.2f} | Combined : {combined_loss:.2f} | LR: {this_lr:.3E} | Memory usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            del loss, combined_loss
            if batch_i % 2000 == 0 and batch_i != 0 and not args.tunelr:
                model.save_to_path(f"/home/ucloud/EUMothModel/models/{run_name}/epoch_{epoch}_batch_{batch_i}.state")
            if batch_i == 100 and run_name == "best_lr":
                break_flag = True
            writer.flush()
            if break_flag:
                break

        # Save checkpoint after each epoch
        if run_name != "best_lr" and not args.val_only and not args.tunelr:
            model.save_to_path(f"/home/ucloud/EUMothModel/models/{run_name}/epoch_{epoch}_batch_final.state")

        time.sleep(1)

        # Evaluate on validation set after each epoch and log metrics to TensorBoard (done after saving checkpoint, to ensure that errors in validation do not prevent checkpoint from being saved)
        model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            
            if not args.tunelr:
                pbar = tqdm(val, total=len(val), desc=f"Validation {epoch}", dynamic_ncols=True)
            else:
                pbar = []
            val_preds = [list() for _ in range(len(model.class_handles["n_classes"]))]
            val_labels = [list() for _ in range(len(model.class_handles["n_classes"]))]
            for batch_i, batch in enumerate(pbar):
                images, labels, paths = batch
                images = images.to(device=device, dtype=dtype)

                wcls = [class_weights[i][labels[i]] for i in range(3)]
                # wcls = [i / i.mean() for i in wcls]

                pred = model(images)
                # Ensure val_pred and val_labels don't get too large
                if len(val_preds) < 1000:
                    for level, i in enumerate(pred):
                        val_preds[level] += pred[level].argmax(-1).cpu().tolist()
                        val_labels[level] += labels[level].int().cpu().tolist()

                del images
                loss = [(loss_fn[i](pred[i], labels[i].to(device)) * wcls[i]).mean() for i in range(3)]
                del pred
                combined_loss = sum([loss[i] * loss_weights[i] for i in range(3)])

                for i, ltype in enumerate(["species", "genera", "families"]):
                    writer.add_scalar(f"Loss/val/{ltype}", loss[i].cpu().float(), pbar.n + len(val) * epoch)
                writer.add_scalar("Loss/val/combined", combined_loss.cpu().float(), pbar.n + len(val) * epoch)

                pbar.set_description_str(f"VAL: Epoch {epoch} | Family : {loss[2]:.2f} | Genus : {loss[1]:.2f} | Species : {loss[0]:.2f} | Combined : {combined_loss:.2f} | Memory usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                del loss, combined_loss
                torch.cuda.empty_cache()
                writer.flush()
                if break_flag:
                    break
            if not args.tunelr:
                confusion_matrix(model, batch, val_preds, val_labels, epoch, writer, tround="val")
        
        if break_flag:
            break
        
        time.sleep(1)
    model.eval()