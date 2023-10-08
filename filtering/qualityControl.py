# This script is used to filter out images of larvae from the dataset.
# OBS: 
# * All images not of types ['JPEG', 'PNG', 'JPG', 'jpeg', 'png', 'jpg'] are ignored.
# * Inference is performed without the alpha channel (if present).
# * Greyscale images are skipped.
# * If an error occurs during inference, the error is logged to a log file and the inference process can be resumed from the last batch by setting the 'skip' variable to the number of batches to skip (Recommended: batch where error occurred minus one, after fixing the error).


# `os` is imported first to ensure that the working directory is set to the root of the project
import os, sys
# Move working directory to the root of the project
os.chdir("/home/ucloud/EUMothModel")
print("Working directory:", os.getcwd())
sys.path.append(os.getcwd())

# Import custom modules for ERDA data transfer
from utils.implicit_mount import *
from utils.dataloader import *

# Import PyTorch modules
import torch
import torch.nn as nn
import torchvision

# Import other modules
from tqdm import tqdm
import traceback

## Helper functions
# Remote file copying helper function
def move_to_dir(paths, dst_dir, client: "IOHandler", verbose=False):
    # Template: mget -d -O sftp://io.erda.au.dk:/AMI_GBIF_Pretraining_Data/NEW_SUPER_DIRECTORY ./PATH_TO_IMAGES
    # The template above is used to copy a file from one location on the remote server to another location on the remote server,
    # while preserving the directory structure of the source location and automatically creating the necessary subdirectories in the destination location.

    cmd_prefix = "mget -d -O sftp://io.erda.au.dk:/AMI_GBIF_Pretraining_Data/"

    cmd_suffix = []
    for path in paths:
        cmd_suffix.append(f"./{path}")
    
    cmd_suffix = " ".join(cmd_suffix)

    cmd = f"{cmd_prefix}{dst_dir} {cmd_suffix}"
    if verbose:
        print("Executing command:", cmd)
    result = client.execute_command(cmd, blocking=True, execute=True)
    if verbose:
        print("Result:", result)

if __name__ == "__main__":

    ## Hyperparameters
    model_name = "qualityControlV1" # Name of model weights file (assumes .pt extension and that the file is located in the models/ directory)
    batch_size = 64 # Batch size for inference 
    skip = 26816 * batch_size # Number of batches to skip (used to resume inference from a specific batch)

    # Device and dtype setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    dtype = torch.bfloat16

    # ERDA data transfer setup
    backend = None
    if backend is not None:
        backend.stop()
    backend = IOHandler(verbose = False, clean=True)
    backend.start()
    backend.cd("AMI_GBIF_Pretraining_Data/root")
    backend.cache_file_index(skip=skip)

    # Model and dataset/loader setup
    # Partial model load, since model preprocessing must be passed to the dataset instance
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    image_preprocessing = weights.transforms(antialias=True)
    def denormalize(tensor, mean=image_preprocessing.mean, std=image_preprocessing.std):
        """Denormalize a tensor."""
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(torch.float32)
        std = torch.tensor(std).view(1, 3, 1, 1).to(torch.float32)
        return tensor.cpu().to(torch.float32) * std + mean

    # Dataset and dataloader setup
    dataset = RemotePathDataset(
        remote_path_iterator=RemotePathIterator(
            backend,
            batch_size=128,
            max_queued_batches=5,
            n_local_files=2*128*5
        ),
        prefetch=3*batch_size,
        transform=image_preprocessing,
        device=device,
        dtype=dtype,
        return_remote_path=True
    )
    dataloader = CustomDataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=6)


    ## Model
    # Model class translation
    QC_dict = {
        0 : "Larvae",
        1 : "Low",
        2 : "High"
    }

    # Model definition
    model = torchvision.models.efficientnet_v2_s(weights = weights).train(False).half()
    num_features = [k for k in [j for j in [i for i in model.children()][0].children()][-1].children()][0].out_channels
    num_classes = 3
    # Model classifier redefinition (**OBS**: THIS MUST MATCH THE MODEL CLASSIFIER DEFINITION IN THE TRAINING SCRIPT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.BatchNorm1d(num_features),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, num_classes),
    )
    # Model checkpoint load
    model.load_state_dict(torch.load(f"models/{model_name}.pt"))
    model = model.to(device=device, dtype=dtype)
    model = model.eval()

    # Check GPU memory usage
    torch.cuda.empty_cache()
    mem_info = [i / 10**9 for i in torch.cuda.mem_get_info(device)]
    print("Memory usage: {:.2f} / {:.2f} GB".format(*mem_info))

    # Create a log file to ensure that the process can be restored if it is interrupted or an error occurs
    assert os.path.isdir("logs"), RuntimeError("Log directory does not exist")

    # Manual logging file
    log_files = os.listdir("logs")
    log_files = [i for i in log_files if i.endswith(".log")]
    log_files = [i for i in log_files if i.startswith(model_name)]
    this_log_file = f"{model_name}_{len(log_files)}.log"

    print("Log file:", this_log_file)

    total_files_transferred = 0

    with open(f"logs/{this_log_file}", "w") as f:
        f.write("")
        # Explicitly disable gradient calculation
        with torch.no_grad():
            # Inference loop
            for i, (input, label, path) in tqdm(enumerate(dataloader), leave = True, total=len(dataloader)):
                # Error handling is logged to the log file instead of being raised
                try:
                    # Model batch inference
                    pred = model(input)
                    # Prediction translation
                    # Good prediction: Larvae % < 5 & High % >= 25
                    # Bad prediction: Larvae % >= 5 & High % < 25

                    # New code
                    # Label translation
                    probs = torch.softmax(pred, dim=1)
                    pcls_idx = torch.argmax(pred, dim=1)
                    l_prob = pred[:, 0]
                    h_prob = pred[:, 2]

                    # QC filtering
                    NOT_Larvae = [p for p, l, h in zip(path, l_prob, h_prob) if l < 0.05 and h >= 0.25]
                    
                    # Move files to 'without_larvae' directory
                    if len(NOT_Larvae) > 0:
                        move_to_dir(NOT_Larvae, "without_larvae", backend)
                        total_files_transferred += len(NOT_Larvae)
                    # Log progress to log file every 100 batches
                    if i % 100 == 0:
                        f.write(f"Batch {i} completed. {total_files_transferred} images moved to 'without_larvae'.\n")
                        f.flush()
                    time.sleep(0.01)
                except Exception as e:
                    # Log and print errors
                    print(f"Error at batch {i}: {e}")
                    tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                    tb_str = ''.join(tb_str)
                    f.write(f"Error at batch {i}: {e}\nStack Trace:\n{tb_str}\n")
                    f.flush()
        f.write(f"Total files transferred: {total_files_transferred}\n")
        f.write("Done.\n")

    # Close the backend
    backend.stop()