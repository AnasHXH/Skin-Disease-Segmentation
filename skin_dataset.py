import os
import random
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from transformers import SamProcessor


def get_bounding_box(ground_truth_map):
    if ground_truth_map.ndim > 2:
        ground_truth_map = np.squeeze(ground_truth_map)
    coords = np.column_stack(np.where(ground_truth_map > 0))
    if coords.size == 0:
        return [0, 0, 0, 0]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def preprocess_for_unet(image_tensor):
    return T.Resize((256, 256))(image_tensor)


class SegmentationAugmentation:
    def __init__(self, resize=(1024, 1024), mask_size=(64, 64), augmentation=True, angle_range=(-10, 10)):
        self.resize = resize
        self.mask_size = mask_size
        self.augmentation = augmentation
        self.angle_range = angle_range if augmentation else (0, 0)

    def __call__(self, image, mask):
        if self.augmentation:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(*self.angle_range)
            image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST, fill=0)
        image = TF.resize(image, self.resize, interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.mask_size, interpolation=T.InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask


class CombinedDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        orig_image = Image.open(self.image_paths[idx]).convert("RGB")
        orig_mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            image_tensor, mask_tensor = self.transform(orig_image, orig_mask)
        else:
            image_tensor = TF.to_tensor(orig_image)
            mask_tensor = TF.to_tensor(orig_mask)

        gt_mask_np = np.array(orig_mask)
        bbox = get_bounding_box(gt_mask_np)
        inputs = self.processor(orig_image, input_boxes=[[bbox]], return_tensors="pt")
        sam_inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        sam_inputs["ground_truth_mask"] = torch.tensor(gt_mask_np, dtype=torch.int64)

        unet_inputs = preprocess_for_unet(image_tensor)
        return sam_inputs, unet_inputs, mask_tensor


def get_file_paths(images_dir, masks_dir, extensions=(".jpg", ".png", ".jpeg"), num_samples=800):
    image_paths = []
    mask_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(images_dir, f"*{ext}")))
        mask_paths.extend(glob(os.path.join(masks_dir, f"*{ext}")))
    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)
    total_samples = min(num_samples, len(image_paths), len(mask_paths))
    indices = random.sample(range(len(image_paths)), total_samples)
    sampled_image_paths = [image_paths[i] for i in indices]
    sampled_mask_paths = [mask_paths[i] for i in indices]
    return sampled_image_paths, sampled_mask_paths
