# This script defines the dataloader for the pretraining dataset.
# The data is contained in a remote server, which is streamed to the local machine.using the implicit_mount module
# The dataset has an inherent hierarchical structure, which is reflected in the directory structure.
# The hierarchy is as follows:
# root
# --Family1
#   --Genus1
#       --Species1
#           img1.jpg/jpeg
#           img2.jpg/jpeg
#           ...
#       --Species2
#           img1.jpg/jpeg
#           img2.jpg/jpeg
#           ...
#       ...
#   --Genus2
#      ...
# --Family2
#   ...

# The classes are the species in the dataset.
# The class labels are the indices of the species, sorted alphabetically, in the dataset.

import os, time

from torchvision.io import read_image
import torch
from torch.utils.data import DataLoader, IterableDataset
from .implicit_mount import RemotePathIterator
from queue import Queue
from threading import Thread

import warnings
from typing import Iterator, List, Tuple, Union

class RemotePathDataset(IterableDataset):
    def __init__(self, remote_path_iterator : "RemotePathIterator", prefetch: int=64, transform=None, target_transform=None, device: Union["torch.device", None]=None, dtype: Union[torch.dtype, None]=None, return_remote_path: bool=False):
        # Check if remote_path_iterator is of type RemotePathIterator
        if not isinstance(remote_path_iterator, RemotePathIterator):
            raise ValueError("Argument remote_path_iterator must be of type RemotePathIterator.")
        # Check if prefetch is an integer
        if not isinstance(prefetch, int):
            raise ValueError("Argument prefetch must be an integer.")
        # Check if prefetch is greater than 0
        if prefetch < 1:
            raise ValueError("Argument prefetch must be greater than 0.")
        
        ## PyTorch specific parameters 
        # Get the classes and their indices
        self.classes = sorted(list(set([path.split('/')[-2] for path in remote_path_iterator.remote_paths])))
        self.n_classes = len(self.classes)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.idx_to_class = {i: self.classes[i] for i in range(len(self.classes))}
        self.return_remote_path = return_remote_path

        # Set the transforms
        self.transform = transform
        self.target_transform = target_transform

        # Set the device and dtype
        self.device = device
        self.dtype = dtype
        
        ## Backend specific parameters
        # Store the remote_path_iterator backend
        self.remote_path_iterator = remote_path_iterator
        self.shuffle = False

        ## Multi-threading parameters
        # Set the number of workers (threads) for (multi)processing
        self.num_workers = 1

        # We don't want to start the buffer filling thread until the dataloader is called for iteration
        self.producer_thread = None
        # We need to keep track of whether the buffer filling thread has been initiated or not
        self.thread_initiated = False

        # Initialize the worker threads
        self.consumer_threads = []
        self.consumers = 0
        
        # Set the buffer filling parameters (Watermark Buffering)
        self.buffer_minfill, self.buffer_maxfill = 0.4, 0.6

        # Initialize the buffers
        self.buffer = Queue(maxsize=prefetch)  # Tune maxsize as needed
        self.processed_buffer = Queue(maxsize=prefetch)  # Tune maxsize as needed

    def _shuffle(self):
        if not self.shuffle:
            raise RuntimeError("Shuffle called, but shuffle is set to False.")
        if self.thread_initiated:
            raise RuntimeError("Shuffle called, but buffer filling thread is still active.")
        self.remote_path_iterator.shuffle()

    def _shutdown_and_reset(self):
        # Handle shutdown logic (Should probably be moved to a dedicated reset function, that is called on StopIteration instead or perhaps in __iter__)
        for i, consumer in enumerate(self.consumer_threads):
            if consumer is None:
                continue
            if not consumer.is_alive():
                continue
            print(f"Waiting for worker {i} to finish.")
            consumer.join(timeout = 1 / self.num_workers) # Wait for the consumer thread to finish
        if self.producer_thread is not None:
            self.producer_thread.join(timeout=1) # Wait for the producer thread to finish
            self.producer_thread = None # Reset the producer thread
        self.consumer_threads = [] # Reset the consumer threads
        self.consumers = 0 # Reset the number of consumers
        self.thread_initiated = False # Reset the thread initiated flag
        self.buffer.queue.clear() # Clear the buffer
        self.processed_buffer.queue.clear() # Clear the processed buffer
        self.remote_path_iterator.__del__()  # Close the remote_path_iterator and clean the temporary directory
        assert self.buffer.qsize() == 0, RuntimeError("Buffer not empty after iterator end.")
        assert self.processed_buffer.qsize() == 0, RuntimeError("Processed buffer not empty after iterator end.")

    def _init_buffer(self):
        # Check if the buffer filling thread has been initiated
        if not self.thread_initiated:
            # Start the buffer filling thread
            self.producer_thread = Thread(target=self._fill_buffer)
            self.producer_thread.daemon = True
            self.producer_thread.start()
            # Set the flag to indicate that the thread has been initiated
            self.thread_initiated = True
        else:
            # Raise an error if the buffer filling thread has already been initiated
            raise RuntimeError("Buffer filling thread already initiated.")

    def _fill_buffer(self):
        # Calculate the min and max fill values for the buffer
        min_fill = int(self.buffer.maxsize * self.buffer_minfill) 
        max_fill = int(self.buffer.maxsize * self.buffer_maxfill)

        print(f"Buffer min fill: {min_fill}, max fill: {max_fill}")

        print("Producer thread started.")
        wait_for_min_fill = False
        for item in self.remote_path_iterator:
            # Get the current size of the buffer
            current_buffer_size = self.buffer.qsize()

            # Decide whether to fill the buffer based on its current size
            if not wait_for_min_fill:
                wait_for_min_fill = (current_buffer_size >= max_fill)

            # Sleep logic which ensures that the buffer doesn't switch between filling and not filling too often (Watermark Buffering)
            while wait_for_min_fill:
                time.sleep(0.1)
                current_buffer_size = self.buffer.qsize() # Update the current buffer size
                wait_for_min_fill = current_buffer_size >= min_fill # Wait until the buffer drops below min_fill

            # Fill the buffer
            self.buffer.put(item)
        
        print("Producer signalling end of iterator.")
        # Signal the end of the iterator to the consumers by putting None in the buffer until all consumer threads have finished
        while self.consumers > 0:
            time.sleep(0.01)
            self.buffer.put(None)
        print("Producer emptying buffer.")
        # Wait for the consumer threads to finish then clear the buffer
        while self.buffer.qsize() > 0:
            self.buffer.get()
        self.processed_buffer.put(None) # Signal the end of the processed buffer to the main thread
        print("Producer thread finished.")
    
    def _process_buffer(self):
        print("Consumer thread started.")
        self.consumers += 1
        while True:
            qsize = self.buffer.qsize()
            while qsize < (self.num_workers * 2):
                time.sleep(0.05 / self.num_workers)
                qsize = self.buffer.qsize()
                if not self.producer_thread.is_alive():
                    break
            # Get the next item from the buffer
            item = self.buffer.get()
            # Check if the buffer is empty, signaling the end of the iterator
            if item is None:
                break  # Close the thread

            # Preprocess the item (e.g. read image, apply transforms, etc.) and put it in the processed buffer
            processed_item = self.parse_item(*item) if item is not None else None
            if processed_item is not None:
                self.processed_buffer.put(processed_item)

        self.consumers -= 1
        print("Consumer thread finished.")

    def __iter__(self):
        # Check if the buffer filling thread has been initiated
        # If it has, reset the dataloader state and close all threads
        # (Only one iteration is allowed per dataloader instance)
        if self.thread_initiated: 
            self._shutdown_and_reset()

        # If shuffle is set to True, shuffle the remote_path_iterator
        if self.shuffle:
            self._shuffle()
        
        # Initialize the buffer filling thread
        self._init_buffer()
        
        # Check number of workers
        if self.num_workers == 0:
            self.num_workers = 1
        if self.num_workers < 1:
            raise ValueError("Number of workers must be greater than 0.")

        # Start consumer threads for processing
        for _ in range(self.num_workers):
            consumer_thread = Thread(target=self._process_buffer)
            consumer_thread.daemon = True
            consumer_thread.start()
            self.consumer_threads.append(consumer_thread)

        return self

    def __next__(self):
        # Fetch from processed_buffer instead
        processed_item = self.processed_buffer.get()

        # Check if the processed buffer is empty, signaling the end of the iterator
        if processed_item is None:
            self._shutdown_and_reset()
            raise StopIteration

        # Otherwise, return the processed item
        return processed_item

    def __len__(self):
        return len(self.remote_path_iterator)
    
    def parse_item(self, local_path, remote_path):
        ## Image processing
        # Check if image format is supported (jpeg/jpg/png)
        image_type = os.path.splitext(local_path)[-1]
        if image_type not in ['.JPG', '.JPEG', '.PNG', '.jpg', '.jpeg', '.png']:
            # raise ValueError(f"Image format of {remote_path} ({image_type}) is not supported.")
            # Instead of raising an error, we can skip the image instead
            return None
        try:
            image = read_image(local_path)
        except:
            print(f"Error reading image {remote_path}.")
            return None
        # Remove the alpha channel if present
        if image.shape[0] == 4:
            image = image[:3]
        if image.shape[0] != 3:
            print(f"Error reading image {remote_path}.")
            return None
        # Apply transforms (preprocessing)
        if self.transform:
            image = self.transform(image)
        if self.dtype is not None:
            image = image.to(dtype=self.dtype)
        if self.device is not None:
            image = image.to(device=self.device)
        
        ## Label processing
        # Get the label by parsing the remote path
        family, genus, species = remote_path.split('/')[-4:-1]
        # TODO: Use the family and genus information (or add a "species only" flag)
        # Transform the species name to the label index
        label = self.class_to_idx[species]
        # Apply label transforms
        if self.target_transform:
            label = self.target_transform(label)
        # TODO: Does the label need to be converted to a tensor and moved to the device? (Probably not) If so, does it need to be a long tensor?
        # if self.device is not None:
        #     label = label.to(device=self.device)

        ## Return the image and label
        if not self.return_remote_path: 
            return image, label
        else:
            return image, label, remote_path

class CustomDataLoader(DataLoader):
    def __init__(self, dataset: "RemotePathDataset", *args, **kwargs):
        # Snipe arguments from the user which would break the custom dataloader (e.g. sampler, shuffle, etc.)
        unsupported_kwargs = ['sampler', 'batch_sampler']
        for unzkw in unsupported_kwargs:
            value = kwargs.pop(unzkw, None)
            if value is not None:
                warnings.warn(f"Argument {unzkw} is not supported in this custom DataLoader. {unzkw}={value} will be ignored.")

        # Override the num_workers argument handling (default is 0)
        dataset.num_workers = kwargs.pop('num_workers', 0)
        # Override the shuffle argument handling (default is False)
        dataset.shuffle = kwargs.pop('shuffle', False)
        
        if not isinstance(dataset, RemotePathDataset):
            raise ValueError("Argument dataset must be of type RemotePathDataset.")

        # Initialize the dataloader
        super(CustomDataLoader, self).__init__(
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            dataset=dataset,
            *args, 
            **kwargs
        )
    
    # TODO: This cannot be set before calling super().__init__(), so perhaps it can be overridden after initialization instead
    # def __setattr__(self, name, value):
    #     if name in ['batch_sampler', 'sampler']:
    #         raise (f"Changing {name} is not allowed in this custom DataLoader.")
    #     super(CustomDataLoader, self).__setattr__(name, value)
