import torch
import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(model, batch, step, writer, tround="val"):
    for level in range(3):
        batch[0] = batch[0].to(device=model.device, dtype=model.dtype)

        with torch.no_grad():
            pred = model(batch[0])
            pcls_probs = pred[level].softmax(-1).cpu().float()
            pcls = pcls_probs.argmax(-1).numpy()
            
            true = batch[1][level].cpu().numpy()

        cm = np.zeros((model.class_handles["n_classes"][level] + 1, model.class_handles["n_classes"][level] + 1))
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
        writer.add_figure(f'{tround}/confmat/{["Species", "Genus", "Family"][level]}', plt.gcf(), global_step=step)
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.hist(pcls_probs.flatten().numpy(), bins=100)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.tight_layout()
        writer.add_figure(f'{tround}/hist/{["Species", "Genus", "Family"][level]}', plt.gcf(), global_step=step)
        plt.close()

    return