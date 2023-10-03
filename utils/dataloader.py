# This script defines the dataloader for the pretraining dataset.
# The data is contained in a remote server, which is mounted to the local filesystem using the mount script.
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
from torch.utils.data import DataLoader, IterableDataset, SequentialSampler
from .implicit_mount import RemotePathIterator
from queue import Queue
from threading import Thread, Lock

import warnings
from typing import Iterator, List, Tuple, Union

class RemotePathDataset(IterableDataset):
    def __init__(self, remote_path_iterator : "RemotePathIterator", prefetch: int=3, transform=None, target_transform=None):
        # Check if remote_path_iterator is of type RemotePathIterator
        if not isinstance(remote_path_iterator, RemotePathIterator):
            raise ValueError("Argument remote_path_iterator must be of type RemotePathIterator.")
        # Check if prefetch is an integer
        if not isinstance(prefetch, int):
            raise ValueError("Argument prefetch must be an integer.")
        # Check if prefetch is greater than 0
        if prefetch < 1:
            raise ValueError("Argument prefetch must be greater than 0.")
        
        # Store the remote_path_iterator backend
        self.remote_path_iterator = remote_path_iterator

        # Initialize the buffers
        self.buffer = Queue(maxsize=prefetch)  # Tune maxsize as needed
        self.processed_buffer = Queue(maxsize=prefetch)  # Tune maxsize as needed

        # We don't want to start the buffer filling thread until the dataloader is called for iteration
        self.thread = None
        # We need to keep track of whether the buffer filling thread has been initiated or not
        self.thread_initiated = False

        # Initialize the worker threads
        self.worker_threads = []

        # Get the classes and their indices
        classes = sorted(list(set([path.split('/')[-2] for path in self.remote_path_iterator])))
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}

        # Set the transforms
        self.transform = transform
        self.target_transform = target_transform
        
        # Set the buffer filling parameters (Watermark Buffering)
        self.buffer_minfill, self.buffer_maxfill = 0.2, 0.8

        # Set the number of workers (threads) for processing
        self.num_workers = 1
        self.lock = Lock()

    def _init_buffer(self):
        # Check if the buffer filling thread has been initiated
        if not self.thread_initiated:
            # Start the buffer filling thread
            self.thread = Thread(target=self._fill_buffer)
            self.thread.daemon = True
            self.thread.start()
            # Set the flag to indicate that the thread has been initiated
            self.thread_initiated = True
        else:
            # Raise an error if the buffer filling thread has already been initiated
            raise RuntimeError("Buffer filling thread already initiated.")

    def _fill_buffer(self):
        # Calculate the min and max fill values for the buffer
        min_fill = int(self.buffer.maxsize * self.buffer_minfill) 
        max_fill = int(self.buffer.maxsize * self.buffer_maxfill)

        for item in self.remote_path_iterator:
            # Get the current size of the buffer
            current_buffer_size = self.buffer.qsize()

            # Decide whether to fill the buffer based on its current size
            should_fill = current_buffer_size < max_fill or current_buffer_size < min_fill

            # Sleep logic which ensures that the buffer doesn't switch between filling and not filling too often (Watermark Buffering)
            while not should_fill:
                time.sleep(0.1)
                current_buffer_size = self.buffer.qsize() # Update the current buffer size )
                should_fill = current_buffer_size < min_fill # Wait until the buffer drops below min_fill

            # Fill the buffer
            self.buffer.put(item)
        
        # Signal the end of the iterator by putting None in the buffer
        self.buffer.put(None)
    
    def _process_buffer(self):
        while True:
            with self.lock:  # Ensure thread-safety when accessing buffers
                # Get the next item from the buffer
                item = self.buffer.get()

            # Preprocess the item (e.g. read image, apply transforms, etc.) and put it in the processed buffer
            processed_item = self.parse_item(*item) if item is not None else None
            with self.lock:  # Ensure thread-safety when accessing buffers
                self.processed_buffer.put(processed_item)

            # Check if the buffer is empty, signaling the end of the iterator
            if item is None:
                if self.thread.is_alive(): # Check if the buffer filling thread is still alive
                    raise RuntimeError("Buffer filling thread is still alive, but buffer signaled end of iteration.")
                self.thread_initiated = False  # Reset the flag to indicate that the thread is no longer active
                break  # Close the thread

    def __iter__(self):
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
            self.worker_threads.append(consumer_thread)

        return self

    def __next__(self):
        # Fetch from processed_buffer instead
        processed_item = self.processed_buffer.get()
        if processed_item is None:
            # Handle shutdown logic
            for worker in self.worker_threads:
                if worker.is_alive():
                    raise RuntimeError("Worker thread is still alive, but processed buffer signaled end of iteration.")
            raise StopIteration

        return processed_item

    def __len__(self):
        return len(self.remote_path_iterator)
    
    def parse_item(self, path, label):
        # Check if image format is supported (jpeg/jpg/png)
        image_type = os.path.splitext(path)[-1]
        if image_type not in ['.jpg', '.jpeg', '.png']:
            raise ValueError(f"Image format of {path} ({image_type}) is not supported.")
        image = read_image(path)
        if self.transform:
            image = self.transform(image)
        family, genus, species = path.split('/')[-4:-1]
        # TODO: Use the family and genus information (or add a "species only" flag)
        label = self.class_to_idx[species]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class SequentialShuffleSampler(SequentialSampler):
    def __init__(self, data_source: "RemotePathDataset"):
        super(SequentialShuffleSampler, self).__init__(data_source)
    
    def __iter__(self):
        self.data_source.remote_path_iterator.shuffle()
        return super(SequentialShuffleSampler, self).__iter__()

class CustomDataLoader(DataLoader):
    def __init__(self, dataset: "RemotePathDataset", *args, **kwargs):
        # Snipe arguments from the user which would break the custom dataloader (e.g. sampler, shuffle, etc.)
        unsupported_kwargs = ['sampler', 'batch_sampler']
        for unzkw in unsupported_kwargs:
            value = kwargs.pop(unzkw, None)
            if value is not None:
                warnings.warn(f"Argument {unzkw} is not supported in this custom DataLoader. {unzkw}={value} will be ignored.")

        # Override the shuffle argument handling (default is False)
        shuffle = kwargs.pop('shuffle', False)
        # Override the num_workers argument handling (default is 0) and pass it to the dataset
        dataset.num_workers = kwargs.pop('num_workers', 0)
        
        if not isinstance(dataset, RemotePathDataset):
            raise ValueError("Argument dataset must be of type RemotePathDataset.")

        # Initialize the dataloader
        super(CustomDataLoader, self).__init__(
            shuffle=False,
            sampler=SequentialSampler(dataset) if not shuffle else SequentialShuffleSampler(dataset), 
            num_workers=0,
            *args, 
            **kwargs)

    def __setattr__(self, name, value):
        if name in ['batch_sampler', 'sampler', 'dataset']:
            raise ValueError(f"Changing {name} is not allowed in this custom DataLoader.")
        super(CustomDataLoader, self).__setattr__(name, value)
