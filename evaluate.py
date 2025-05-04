import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import compute_dice, compute_iou, pixel_accuracy
from losses import combined_loss


def evaluate_model(model, datasets_to_test, checkpoint_path, collate_fn=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model.to(device)

    for dataset_name, dataset in datasets_to_test.items():
        print(f"\n--- Evaluating {dataset_name} dataset ---")
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        metrics = {'loss': 0, 'acc': 0, 'dice': 0, 'iou': 0}
        total_samples = 0

        loop = tqdm(loader, desc=f"Evaluating {dataset_name}", leave=False)
        for sam_inputs, unet_inputs, masks in loop:
            batch_size = unet_inputs.size(0)
            total_samples += batch_size

            sam_tensor = sam_inputs["pixel_values"].to(device)
            unet_tensor = unet_inputs.to(device)
            if isinstance(masks, list):
                masks = torch.stack(masks, dim=0)
            masks = masks.to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(sam_tensor, unet_tensor)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    pred_mask = (outputs > 0.5).float()

            loss = combined_loss(outputs, masks)
            metrics['loss'] += loss.item() * batch_size
            metrics['acc'] += pixel_accuracy(pred_mask, masks) * batch_size
            metrics['dice'] += compute_dice(pred_mask, masks) * batch_size
            metrics['iou'] += compute_iou(pred_mask, masks) * batch_size

            loop.set_postfix({
                'loss': f"{metrics['loss'] / total_samples:.4f}",
                'acc': f"{metrics['acc'] / total_samples:.4f}",
                'dice': f"{metrics['dice'] / total_samples:.4f}",
                'iou': f"{metrics['iou'] / total_samples:.4f}"
            })

        print(f"{dataset_name} Final Metrics: Loss={metrics['loss'] / total_samples:.4f}, "
              f"Acc={metrics['acc'] / total_samples:.4f}, Dice={metrics['dice'] / total_samples:.4f}, "
              f"IoU={metrics['iou'] / total_samples:.4f}")
