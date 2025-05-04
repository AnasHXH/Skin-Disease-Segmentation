import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
from sam_unet import SAM_U_Net_Model
from metrics import compute_dice, compute_iou, pixel_accuracy, combined_loss


def train_model(train_loader, val_loader, sam_model_registry, checkpoint_path, train_dataset, val_dataset, num_epochs=100, patience=10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SAM_U_Net_Model(
        out_channels=1,
        sam_model_registry=sam_model_registry,
        checkpoint_path=checkpoint_path,
        use_patch_embed=True,
        bilinear=True,
        pretrained=True
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    model = model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.1 + 0.9 * min(1.0, epoch / 5)
    )

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    early_stop_counter = 0

    history = {
        'train_loss': [], 'train_acc': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_acc': [], 'val_dice': [], 'val_iou': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_metrics = {'loss': 0, 'acc': 0, 'dice': 0, 'iou': 0}
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        optimizer.zero_grad()
        scaler = amp.GradScaler()

        for sam_inputs, unet_inputs, masks in loop:
            sam_tensor = sam_inputs["pixel_values"].to(device, memory_format=torch.channels_last)
            unet_inputs = unet_inputs.to(device, memory_format=torch.channels_last)
            masks = masks.to(device, memory_format=torch.channels_last).long()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(sam_tensor, unet_inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = combined_loss(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_size = masks.size(0)
            train_metrics['loss'] += loss.item() * batch_size
            train_metrics['acc'] += pixel_accuracy(outputs, masks) * batch_size
            train_metrics['dice'] += compute_dice(outputs, masks) * batch_size
            train_metrics['iou'] += compute_iou(outputs, masks) * batch_size

        n = len(train_dataset)
        for k in train_metrics:
            train_metrics[k] /= n
            history[f'train_{k}'].append(train_metrics[k])

        model.eval()
        val_metrics = {'loss': 0, 'acc': 0, 'dice': 0, 'iou': 0}
        with torch.no_grad():
            for sam_inputs, unet_inputs, masks in val_loader:
                sam_tensor = sam_inputs["pixel_values"].to(device)
                unet_inputs = unet_inputs.to(device)
                masks = masks.to(device)

                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(sam_tensor, unet_inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = combined_loss(outputs, masks)

                batch_size = masks.size(0)
                val_metrics['loss'] += loss.item() * batch_size
                val_metrics['acc'] += pixel_accuracy(outputs, masks) * batch_size
                val_metrics['dice'] += compute_dice(outputs, masks) * batch_size
                val_metrics['iou'] += compute_iou(outputs, masks) * batch_size

        n_val = len(val_dataset)
        for k in val_metrics:
            val_metrics[k] /= n_val
            history[f'val_{k}'].append(val_metrics[k])

        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss={train_metrics['loss']:.4f}, Val Loss={val_metrics['loss']:.4f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "SEG_best_model.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping.")
                break

        torch.cuda.empty_cache()

    return model, history
