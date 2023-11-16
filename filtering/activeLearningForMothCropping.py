# This script is used to create a small fine-tuning dataset for salient/non-salient moth localization using YOLOv5 and an active learning approach.
# The script will load a pre-trained YOLOv5 model and run inference on a dataset of moth images. The user will then be able to correct the bounding boxes
# and save the corrected bounding boxes to a new dataset for fine-tuning. The script will also create a YAML file for the fine-tuning dataset.
# The script will also create a log file with the indices and splits for the fine-tuning dataset. This log file can be used to resume the script from a specific
# batch in case the script is interrupted, or if the user wants to correct bounding boxes in smaller batches.

import os

# Import other modules
from tqdm import tqdm
import numpy as np
import h5py
import time

from PIL import Image

import dash
from dash import dcc, html, Output, Input, State
import plotly.express as px

from matplotlib import pyplot as plt

from multiprocessing import Process, Manager

import torch

# Import YOLOv5 helper functions
os.chdir("/home/ucloud/EUMothModel")
from tutils.yolo_helpers import non_max_suppression

model_weights = "insect_iter6-1280m7.pt"
model = torch.hub.load(
    'ultralytics/yolov5', 
    'custom', 
    path=f'models/{model_weights}',
    force_reload=False)

import random

## Hyperparameters
skip = 0 # Number of batches to skip (used to resume script from a specific batch)
batch_size = 8 # Batch size for chunked loading of images
n_batches = 1 # Number of batches to load
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype=torch.float16
real_class_index = 0 # The real class index of the species to fine-tune on (6 for the initial model; insectGBIF-1280m6.pt)
inference_size = 640 # The size of the images to run inference on (1280 for the initial model; insectGBIF-1280m6.pt)


# HDF5 Dataset Class
class HDF5Tensors:
    def __init__(self, dataset: "h5py.Dataset", length: int):
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f'`dataset` must be an instance of "h5py.Dataset" not "{type(dataset)}"') 
        self.dataset = dataset
        self.index = 0
        self.length = length
        # Channels, Height, Width
        self.shape = self.dataset.shape[1:]
        self.channels, self.height, self.width = self.shape

    def __getitem__(self, index):
        if isinstance(index, slice):
            return torch.from_numpy(self.dataset[index])
        else:
            return torch.from_numpy(self.dataset[index])
    
    def __len__(self):
        return self.length
    
    def __repr__(self):
        return f'HDF5Tensors({self.dataset})'
    
    def __str__(self):
        return self.__repr__()
    
    def __iter__(self):
        return self.__init__(self.dataset, self.length)
    
    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        else:
            self.index += 1
            return self[self.index - 1]     

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, hdf5file, tensorname, classname, *args, **kwargs):
        self.hdf5 = h5py.File(hdf5file, 'r', rdcc_nslots=8191, rdcc_nbytes=128 * 1024**2, rdcc_w0=0.75, swmr=True)
        self.classes = self.hdf5[classname]
        self.class_set = []
        for i in tqdm(range(0, len(self.classes), 1000), desc="Building class set", unit="chunks", leave=False):
            self.class_set += list(self.classes[i:i+1000])
        self.length = len(self.class_set)
        self.class_set = sorted(list(set(self.class_set)))
        self.n_classes = len(self.class_set)
        self.class_to_idx = {self.class_set[i]: i for i in range(len(self.class_set))}
        self.idx_to_class = {i: self.class_set[i] for i in range(len(self.class_set))}

        self.tensors = HDF5Tensors(self.hdf5[tensorname], self.length)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.tensors[index], [self.class_to_idx[i] for i in self.classes[index]]
        elif isinstance(index, np.ndarray) or isinstance(index, torch.Tensor) or isinstance(index, list):
            if isinstance(index, torch.Tensor):
                index = index.numpy()
            if isinstance(index, list):
                types = set([type(i) for i in index])
                if len(types) > 1:
                    raise TypeError(f'`index` must be a list of a single type, not {types}')
                single_type = types.pop()
                if not single_type in [int, bool]:
                    raise TypeError(f'`index` must be a list of integers or booleans, not {single_type}')
                index = np.array(index)
            if len(index) == 0:
                raise IndexError("Empty index")
            if not index.dtype == np.int64 and not index.dtype == np.bool_:
                print(f'Warning: Converting index from {index.dtype} to int64 for indexing HDF5 dataset!')
                index = index.astype(np.int64)
            if index.dtype == np.bool_:
                if len(index) != self.length:
                    raise IndexError(f'`index` must be a boolean array of length {self.length}, not {len(index)}')
                index = np.where(index)[0]
            if (index.max() > self.length):
                raise IndexError(f'`index` out of range: {index.max()} > {self.length}')
            if (index.min() < 0):
                raise IndexError(f'`index` out of range: {index.min()} < 0')
            return self.tensors[index], [self.class_to_idx[i] for i in self.classes[index]]
        else:
            return self.tensors[index], self.class_to_idx[self.classes[index]]
        
    
    def __len__(self):
        return self.length

### Dash app

# Initialize your app
styles = [
    #<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
    {
        'href': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'intregrity': 'sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN',
        'crossorigin': 'anonymous'
    }
]

scripts = [
    #<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL" crossorigin="anonymous"></script>
    {
        'src': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js',
        'integrity': 'sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL',
        'crossorigin': 'anonymous'
    }
]


app = dash.Dash(__name__, external_stylesheets=styles, external_scripts=scripts)
app.config.suppress_callback_exceptions = False

# Define the app state
manager = Manager()
corrected_boxes = manager.list([])
boxes = None  # DONT CHANGE THIS
height, width = None, None # DONT CHANGE THIS

# Define the app callbacks
@app.callback(
    Output('graph', 'figure'), # The figure is passed to the graph object to update the image and bounding boxes
    [Input('next-dummy', 'n_clicks')], # The next-dummy object is used as a trigger to avoid circular callbacks.
    [State('next-button', 'n_clicks'), # Which image of the currently processed images to show
     State('graph', 'relayoutData'), # Not used at the moment
     State('graph', 'figure')] # Not used at the moment
)
def update_figure(n0, n, relayoutData, existing):
    global height, width
    if n is None:
        n = 0  # Initialize
    if n >= len(indices):  # Replace with your batch_size
        raise SystemExit("Stopping Dash app.")
    
    img = dataset[indices[n]][0].numpy().transpose(1, 2, 0)  # Replace with your image data
    # Upscale image to 1280x1280 with numpy (img is a ndarray)
    # img = np.array(Image.fromarray(img).resize((1280, 1280), resample=Image.BILINEAR))
    height, width, _ = img.shape
    initial_boxes = boxes[n]  # Replace with your initial boxes for image n

    fig = px.imshow(img, origin='upper', height=750 * height/width, width=750)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    shapes = []
    if len(initial_boxes) > 0:
        for box in initial_boxes:
            x1, y1, x2, y2 = box[:4]
            x1 *= width/inference_size
            x2 *= width/inference_size
            y1 *= height/inference_size
            y2 *= height/inference_size
            if x1 == x2 or y1 == y2:
                continue
            shapes.append(
                {'x0': x1, 'y0': y1, 'x1': x2, 'y1': y2, 'editable': True}
            )
        
        if len(shapes) > 0:
            fig.update_layout(
                shapes=shapes,
                dragmode='drawrect',
                newshape=dict(line_color='cyan')
            )

    return fig

@app.callback(
    Output('next-dummy', 'n_clicks'), # The dummy button is used to trigger the callback only, resulting in update_figure being called.
    [Input('next-button', 'n_clicks')], # The button is used to trigger the callback, i.e. boxes are saved when the 'Next' button is clicked.
    [State('store', 'data'), # The store doesn't do anything at the moment.
     State('graph', 'figure')] # The figure is used to get the shape data (bounding boxes)
)
def update_store(n, existing_data, figure):
    if n is None:
        n = 0
    else:
        n = n - 1

    while len(corrected_boxes) <= n:
        corrected_boxes.append([])

    if figure and 'layout' in figure and 'shapes' in figure['layout'] and len(figure['layout']['shapes']) > 0:
        new_list = []
        for shape_data in figure['layout']['shapes']:
            x0, y0, x1, y1 = shape_data['x0'], shape_data['y0'], shape_data['x1'], shape_data['y1']
            x0 *= inference_size/width
            x1 *= inference_size/width
            y0 *= inference_size/height
            y1 *= inference_size/height
            new_box = [x0, y0, x1, y1]
            print(f'New box: {new_box}')
            new_list.append(new_box)
        corrected_boxes[n] = new_list
    else:
        corrected_boxes[n] = []
    
    print(f'Number of corrected boxes: {len(corrected_boxes)}')

    return n

# This callback is used to test the responsiveness of the app. It will print the number of times the test button has been clicked.
@app.callback(
    Output('test-store', 'data'), 
    [Input('test-button', 'n_clicks')], 
    [State('store', 'data'), 
     State('graph', 'figure')]
)
def test_responsiveness(n, existing_data, figure):
    print(f'Test button clicked {n} times')
    return n

# Define the app layout
app.layout = html.Div([
    dcc.Graph(
        id='graph',
        config={
            'modeBarButtonsToAdd': ['drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['zoom2d', 'zoomIn2d', 'zoomOut2d', 'pan2d', 'autoScale2d', 'resetScale2d', 'toImage']
        }
    ),
    html.Div(
        id='Controls', 
        children=[
            html.Button('Next', id='next-button', className='btn btn-primary', style={'margin': '10px'}),
            html.Button('Test', id='test-button', className='btn btn-primary', style={'margin': '10px'})
        ],
        style={
            'display': 'flex', 
            'justify-content': 'center',
            'align-items': 'center',
            'margin-top': '10px',
            'margin-bottom': '10px',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'width': '100%',
            'height': '50px'
        }
    ),
    html.Div(id='next-dummy', style={'display': 'none'}, n_clicks=0),
    dcc.Store(id='test-store'),
    dcc.Store(id='store')
])

if __name__ == "__main__":
    
    dataset = HDF5Dataset("datasets/rebalanced75_without_larvae.h5", "images", "species")

    length = len(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )

    model = model.to(device,dtype=dtype)

    # Test the model
    model.eval()

    # Filter already processed indices
    processed_indices = []
    if os.path.isfile("datasets/fine_tuning.log"):
        with open("datasets/fine_tuning.log", "r") as f:
            for line in f:
                i, split = line.split(" ")
                processed_indices.append(int(i))

    indices = []
    predictions = []

    for batch in tqdm(range(n_batches + int(len(processed_indices) / batch_size)), desc="Running inference", unit="batches", leave=False):
        print(f'Attempting to process batch {batch} of {n_batches}')
        if batch >= n_batches:
            break
        # Sample `batch_size` indices from the dataset
        i = random.sample(range(length), batch_size)
        i = np.array(i)
        i.sort()
        # Skip indices that have already been processed
        i = i[~np.isin(i, processed_indices)]
        if len(i) == 0:
            n_batches += 1
            print("skipping batch")
            continue
        print(i)
        indices += list(i)
        input, labels = dataset[i]
        assert len(input.shape) == 4, f"Input shape is {input.shape}, should be (n, c, h, w)"
        input = input.to(device, dtype=torch.float32)
        input = torch.nn.functional.interpolate(input, (inference_size, inference_size), mode="bilinear", align_corners=True, antialias=True).to(dtype=dtype)
        pred = model(input / 255, augment=True)
        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = [i for i in pred if i is not None]
            pred = torch.cat(pred, dim=1) # TODO: Why does this even work?
        pred = pred.clone().to(torch.float32)
        # Filter prediction == 5 by setting confidence to 0 for bounding boxes with argmax class 5
        pred[..., 4][torch.argmax(pred[..., 5:], dim=-1) != real_class_index] = -1
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0, max_det=5) # list of detections, on (n,6) tensor per image [xyxy, conf, cls]

        predictions += pred

    # Select best bounding box for each image (if any)
    best_bbox = []
    for i in predictions:
        i = i.cpu()
        if i.shape[0] == 0:
            best_bbox += [torch.zeros(1, 6)]
            continue
        try:
           best_bbox += [i[i[:, 4] == i[:, 4].max()]]
        except Exception as e:
            print(f"Error on selecting best bounding box from {i}")
            raise e
        
    del predictions, pred, input, labels, batch, model, dataloader
    torch.cuda.empty_cache()

    boxes = torch.stack(best_bbox).numpy()

    # Plot the images with bounding boxes
    # result = plot_images_with_boxes(images, best_bbox)
    def run_dash_app():
        app.run(debug=False, dev_tools_ui=False, dev_tools_props_check=False, dev_tools_hot_reload=False, threaded=False)

    p = Process(target = run_dash_app)
    p.start()

    # Wait for the app to stop
    while p.is_alive():
        time.sleep(0.1)
    
    corrected_boxes = list(corrected_boxes) # Convert from Manager.list to list

    # print("Corrected boxes:")
    # print(corrected_boxes)

    # # Plot the images with bounding boxes
    # fig, axes = plt.subplots(len(images), 1, figsize=(10, 10 * len(images)))
    # for i, ax in enumerate(axes.flatten()):
    #     ax.axis("off")
    #     ax.set_aspect("equal")
    #     ax.imshow(images[i].transpose(1, 2, 0))
    #     for box in corrected_boxes[i]:
    #         x1, y1, x2, y2 = box
    #         ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="cyan", linewidth=2))
    #     x1, y1, x2, y2 = boxes[i][0][:4]
    #     ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2))

    # fig.savefig("logs/corrected_boxes.png", bbox_inches="tight", pad_inches=0)
    # plt.close()

    
    ## Create YOLOv5 fine tuning dataset
    # Create the YOLOv5 compatible folder structure
    if not os.path.isdir("datasets/fine_tuning"):
        os.makedirs("datasets/fine_tuning", exist_ok=False)
    if not os.path.isdir("datasets/fine_tuning/images"):
        os.makedirs("datasets/fine_tuning/images", exist_ok=False)
    if not os.path.isdir("datasets/fine_tuning/labels"):
        os.makedirs("datasets/fine_tuning/labels", exist_ok=False)
    for t in ["train", "valid", "test"]:
        if not os.path.isdir(f"datasets/fine_tuning/images/{t}"):
            os.makedirs(f"datasets/fine_tuning/images/{t}", exist_ok=False)
        if not os.path.isdir(f"datasets/fine_tuning/labels/{t}"):
            os.makedirs(f"datasets/fine_tuning/labels/{t}", exist_ok=False)

    # Create log file for fine tuning (ONLY if it doesn't exist)
    if not os.path.isfile("datasets/fine_tuning.log"):
        with open("datasets/fine_tuning.log", "w") as f:
            pass

    # Indices type is a list of strings either "train", "valid" or "test" depending on the split for each index in `indices`
    indices_type = []
    # Create train, valid and test splits
    for i in tqdm(indices, desc="Creating train, valid and test splits", unit="images", leave=False):
        if i % 10 == 0:
            indices_type.append("test")
        elif i % 10 == 1:
            indices_type.append("valid")
        else:
            indices_type.append("train")
    
    # Save images and labels to disk for fine tuning and update the log file
    log = open("datasets/fine_tuning.log", "a")
    # Loop over all processed indices (corresponding to images in the HDF5 dataset) and save the images and bounding boxes to disk
    for i, (ind, boxs) in tqdm(enumerate(zip(indices, corrected_boxes)), total=len(indices), desc="Creating fine tuning dataset", unit="images", leave=False):
        # Get the split for the current index/image
        split = indices_type[i]

        # Load image from HDF5 dataset using the index
        img = dataset[ind][0].numpy().transpose(1, 2, 0)
        # Save image to disk
        Image.fromarray(img).save(f"datasets/fine_tuning/images/{split}/{ind}.jpg")

        # Create label file for the current image
        with open(f"datasets/fine_tuning/labels/{split}/{ind}.txt", "w") as f:
            # Find the salient box (heuristic: largest box)
            areas = []
            for box in boxs:
                x1, y1, x2, y2 = box
                areas.append((x2 - x1) * (y2 - y1))
            salient_box = np.argmax(areas)

            # Loop over all boxes and save them to the label file, 
            # with the salient box having class 0 and all other boxes having class 1
            for j, box in enumerate(boxs):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = x1 / inference_size, y1 / inference_size, x2 / inference_size, y2 / inference_size
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1), abs(y2 - y1)
                cls = 0 if j == salient_box else 1
                f.write(f"{cls} {x} {y} {w} {h}\n")
        
        # Write the index and split to the log file
        log.write(f"{ind} {split}\n")
    # Flush and close the log file
    log.flush()
    log.close()
        
    # Create YAML file for fine tuning (unnecessarily overrides the file if it already exists, in case the script is run multiple times, but it doesn't matter)
    with open("datasets/fine_tuning.yaml", "w") as f:
        f.write("Fine-tuning dataset YAML file\n")
        f.write("\n")
        f.write("# Content:\n")
        f.write("path: ~/datasets/fine_tuning\n")
        f.write("train: images/train\n")
        f.write("val: images/valid\n")
        f.write("test: images/test\n")
        f.write("\n")
        f.write("# Classes:\n")
        f.write("nc: 2\n")
        f.write("names: ['salient_moth', 'non_salient_moth']\n")
        