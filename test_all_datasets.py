import os
import torch
from torch.utils.data import DataLoader
import random
from skin_dataset import CombinedDataset, SegmentationAugmentation, get_bounding_box, preprocess_for_unet
from evaluate import evaluate_model
from transformers import SamProcessor
from sam_unet import SAM_U_Net_Model

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset paths
datasets = {
    "ISIC-2016": {
        "images": "/kaggle/input/isic-segmentation-2016/ISIC 2016/test",
        "masks": "/kaggle/input/isic-segmentation-2016/ISIC 2016/test_masks"
    },
    "ISIC-2018": {
        "images": "/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input",
        "masks": "/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth"
    },
    "HAM10000": {
        "images": "/kaggle/input/ham1000-segmentation-and-classification/images",
        "masks": "/kaggle/input/ham1000-segmentation-and-classification/masks"
    }
}

# Load the SAM processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
augmentation = SegmentationAugmentation(resize=(1024, 1024), mask_size=(64, 64), augmentation=False)

# Helper to get file paths
from glob import glob
from PIL import Image
import numpy as np

def get_file_paths(images_dir, masks_dir, extensions=(".jpg", ".png", ".jpeg"), num_samples=800):
    image_paths = []
    mask_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(images_dir, f"*{ext}")))
        mask_paths.extend(glob(os.path.join(masks_dir, f"*{ext}")))
    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)
    total_samples = min(num_samples, len(image_paths), len(mask_paths))
    indices = random.sample(range(total_samples), total_samples)
    sampled_image_paths = [image_paths[i] for i in indices]
    sampled_mask_paths = [mask_paths[i] for i in indices]
    return sampled_image_paths, sampled_mask_paths

# Custom collate for SAM inputs
def custom_collate_fn(batch):
    sam_inputs_list, unet_inputs_list, mask_tensor_list = zip(*batch)
    collated_sam_inputs = {}
    for key in sam_inputs_list[0].keys():
        shapes = [d[key].shape for d in sam_inputs_list]
        if all(shape == shapes[0] for shape in shapes):
            collated_sam_inputs[key] = torch.stack([d[key] for d in sam_inputs_list], dim=0)
        else:
            collated_sam_inputs[key] = [d[key] for d in sam_inputs_list]
    collated_unet_inputs = torch.stack(unet_inputs_list, dim=0)
    if all(m.shape == mask_tensor_list[0].shape for m in mask_tensor_list):
        collated_masks = torch.stack(mask_tensor_list, dim=0)
    else:
        collated_masks = list(mask_tensor_list)
    return collated_sam_inputs, collated_unet_inputs, collated_masks

# Prepare all datasets
datasets_to_test = {}
for name, paths in datasets.items():
    imgs, masks = get_file_paths(paths["images"], paths["masks"])
    dataset = CombinedDataset(imgs, masks, processor, transform=augmentation)
    datasets_to_test[name] = dataset

# Initialize model
sam_model_registry = {
    "vit_h": lambda checkpoint: torch.hub.load("facebookresearch/segment-anything", "sam_vit_h", checkpoint=checkpoint)
}

model = SAM_U_Net_Model(
    out_channels=1,
    sam_model_registry=sam_model_registry,
    checkpoint_path="sam_vit_h.pth",
    use_patch_embed=True,
    bilinear=True,
    pretrained=True
)

# Evaluate all datasets
evaluate_model(model, datasets_to_test, checkpoint_path="./checkpoints/SEG_best_model.pth", collate_fn=custom_collate_fn)
