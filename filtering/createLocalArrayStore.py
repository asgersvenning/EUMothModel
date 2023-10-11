# This script is used to create a local array store of the dataset, which can be used to perform inference on the dataset.
# I will use np.memmap to create the array store, since it allows for lazy loading of the data, which is necessary due to the size of the dataset.

# `os` is imported first to ensure that the working directory is set to the root of the project
import os, sys

# Move working directory to the root of the project
os.chdir("/home/ucloud/EUMothModel")
print("Working directory:", os.getcwd())
sys.path.append(os.getcwd())

# Import custom modules for ERDA data transfer
from tutils.implicit_mount import *
from tutils.dataloader import *

# Import other modules
from tqdm import tqdm
import numpy as np
import hdf5plugin
import h5py

import torch

# # DEBUGGING
# from matplotlib import pyplot as plt
# from torchvision.io import read_image

## Hyperparameters
dtype = torch.uint8 # Datatype of the array store
np_dtype = np.uint8 # Datatype of the array store
target_resolution = (384, 384) # Resolution of the images in the array store
skip = 21961*64 # Number of batches to skip (used to resume script from a specific batch)
batch_size = 64 # Batch size for chunked loading of images
device = torch.device("cpu") # CPU is used since this script merely creates the array store

# Helper functions
def image_preprocessing(image: "torch.Tensor") -> "torch.Tensor":
    """
    Preprocess an image tensor.

    Args:
        image (torch.Tensor): Image tensor to preprocess. Should be of shape (3, H, W).
    
    Returns:
        torch.Tensor: Preprocessed image tensor. Will be of shape (3, target_resolution[0], target_resolution[1]) and dtype `dtype`.
    """
    image = image.unsqueeze(0)

    # Resize image
    image = torch.nn.functional.interpolate(image, target_resolution, mode="bilinear", align_corners=False, antialias=True)

    return image.squeeze(0)

if __name__ == "__main__":
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
    dataloader = CustomDataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    print(len(dataset))

    # # DEBUGGING
    # total_disk_size = 0
    # original_images = []

    # Fill array store
    with h5py.File("datasets/rebalanced75_without_larvae.h5", "r+") as f:
        # f.create_dataset("images", chunks=(1, 3, 384, 384), shape=(len(dataset), 3, *target_resolution), dtype=np_dtype, **hdf5plugin.SZ(absolute=5))
        # f.create_dataset("species", shape=(len(dataset),), dtype=h5py.string_dtype(encoding="utf-8"))
        # BEST SPEED+COMPRESSION: **hdf5plugin.Blosc2(cname="zstd", clevel=3, filters=hdf5plugin.Blosc2.DELTA)) # 1.5X compression, >37 images/s
        # BEST COMPRESSION: **hdf5plugin.SZ(absolute=5) # 3X compression, >16 images/s
        
        for i, (images, cls, paths) in enumerate(tqdm(dataloader, desc="Writing to local database", dynamic_ncols=True)):
            # Write to hd5 file
            n = len(images)
            ir = i + skip
            f["images"][ir*batch_size:(ir*batch_size + n)] = images
            f["species"][ir*batch_size:(ir*batch_size + n)] = [dataset.idx_to_class[c.item()] for c in cls]

            if i % 10 == 0:
                f.flush()

            # # DEBUGGING
            # original_images += paths
            #
            # for path in paths:
            #     total_disk_size += os.path.getsize(path)

    # # DEBUGGING
    # # Print the size of the dataset on disk in MB
    # print(f"Dataset size: {total_disk_size / 1e6:.1f} MB")

    # # # Print the size of the array store in MB
    # # array_store_size = os.path.getsize('datasets/array_store_rebalanced75_without_larvae')
    # # print(f"Array store size: {array_store_size / 1e6:.1f} MB")

    # # # Print the compression ratio
    # # print(f"Array Store Compression ratio: {total_disk_size / array_store_size:.2f}X")

    # # Print the size of the hd5 file in MB
    # hdf5_size = os.path.getsize('datasets/rebalanced75_without_larvae.h5')
    # print(f"HDF5 size: {hdf5_size / 1e6:.1f} MB")

    # # Print the compression ratio
    # print(f"HDF5 Compression ratio: {total_disk_size / hdf5_size:.2f}X")

    # # Print some of the images in the hdf5-dataset alongside the original images to inspect any potential artifacts
    # plt.figure(figsize=(50, 10))

    # with h5py.File("datasets/hdf5_rebalanced75_without_larvae", "r") as f:
    #     for i in range(5):
    #         or_img = read_image(original_images[i]).permute(1, 2, 0).numpy()
    #         or_img = or_img / 255

    #         plt.subplot(2, 10, i+1)
    #         plt.imshow(or_img)
    #         plt.text(0, 0, f"Original image {i}", color="white", fontsize=12, fontweight="bold", bbox=dict(facecolor='black', alpha=0.5))
    #         plt.axis('off')

    #         # Image is already a ndarray
    #         hdf5_img = f["images"][i]
    #         # "Permute"
    #         hdf5_img = hdf5_img.transpose(1, 2, 0)
    #         hdf5_img = hdf5_img / 255

    #         plt.subplot(2, 10, i+11)
    #         plt.imshow(hdf5_img)
    #         plt.text(0, 0, f"HDF5 image {i}", color="white", fontsize=12, fontweight="bold", bbox=dict(facecolor='black', alpha=0.5))
    #         plt.axis('off')

    # plt.savefig("logs/hdf5_test.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # Close backend
    backend.stop()


