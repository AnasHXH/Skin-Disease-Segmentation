import torch
import torch.nn as nn
import torch.nn.functional as F
import monai

bce_loss_fn = nn.BCEWithLogitsLoss()

def prepare_target(target):
    if target.dim() == 4 and target.shape[1] == 1:
        return target.squeeze(1)
    return target

def prepare_bce_targets(targets, num_classes):
    if num_classes == 1:
        targets = prepare_target(targets)
        return targets.unsqueeze(1).float() if targets.dim() == 3 else targets.float()
    else:
        return monai.networks.utils.one_hot(targets, num_classes).float()

def iou_loss(inputs, targets, smooth=1e-6):
    if inputs.shape[1] == 1:
        inputs = torch.sigmoid(inputs)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
    else:
        inputs = torch.softmax(inputs, dim=1)

    intersection = (inputs * targets).sum(dim=(2, 3))
    union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    loss = -torch.log(iou + smooth)
    return loss.mean()

def combined_loss(outputs, targets, w_iou=0.5, w_bce=0.5):
    if outputs.shape[1] != 1:
        raise ValueError("This combined_loss function is for binary segmentation.")

    targets_iou = targets if targets.dim() == 4 else targets.unsqueeze(1)
    if targets_iou.shape[2:] != outputs.shape[2:]:
        targets_iou = F.interpolate(targets_iou.float(), size=outputs.shape[2:], mode='nearest')
    loss_iou_val = iou_loss(outputs, targets_iou)

    targets_bce = prepare_bce_targets(targets, 1)
    if targets_bce.shape[2:] != outputs.shape[2:]:
        targets_bce = F.interpolate(targets_bce, size=outputs.shape[2:], mode='nearest')
    loss_bce_val = bce_loss_fn(outputs, targets_bce)

    return w_iou * loss_iou_val + w_bce * loss_bce_val
