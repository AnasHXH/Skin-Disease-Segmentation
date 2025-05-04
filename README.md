# Skin-Disease-Segmentation

**Skin Disease Segmentation and Harmonizing Global and Local Features via Hyper-Attention Fusion U-Net with SAM Integration**

---

## Overview
This repository provides the implementation for our proposed architecture, **H-Fusion SEG**, designed for precise skin disease segmentation. The model integrates the **Segment Anything Model (SAM)** and **U-Net** via a **Hyper-Attention Fusion mechanism**, enabling robust global context understanding while preserving fine-grained lesion details.

The framework has been tested on benchmark datasets:
- ISIC-2016
- ISIC-2018
- HAM10000

---

## Project Structure
```bash
├── sam_unet.py                     # Model architecture (SAM + U-Net + Hyper Attention)
├── skin_dataset.py                 # Dataset preparation and augmentation
├── losses.py                       # Loss functions and evaluation metrics
├── evaluate.py                     # Evaluation and visualization functions
├── test_all_datasets.py           # Script to test the model on all benchmark datasets
├── train.py                        # Training pipeline (Coming soon)
├── requirements.txt                # Required packages
├── README.md                       # Project documentation
├── Test_and_Evaulation.ipynb.ipynb # Jupyter notebook to view output results visually
├── ISIC-2016 dataset.png           # Output samples from ISIC-2016
├── ISIC-2018 dataset.png           # Output samples from ISIC-2018
├── HAM 10000 dataset.png           # Output samples from HAM10000
└── checkpoints/                    # Folder to store saved models (to be created)
```

---

## Installation
```bash
pip install -r requirements.txt
```
> Note: Make sure to have a CUDA-compatible GPU and install the [Segment Anything](https://github.com/facebookresearch/segment-anything) dependencies.

---

## Training
> **Coming Soon**

---

## Evaluation
To evaluate the model on all datasets:
```bash
python test_all_datasets.py
```
This will run inference and print metrics (Loss, Accuracy, Dice, IoU) for each dataset.

---

## Output Samples
The model outputs include:
- Input Image
- Ground Truth Mask
- Predicted Mask
- Segmented Image (Masked Output)
- Metrics (Loss, Accuracy, Dice, IoU)

### 📌 Sample Results:
![ISIC 2016 Sample](ISIC-2016%20dataset.png)
![ISIC 2018 Sample](ISIC-2018%20dataset.png)
![HAM 10000 Sample](HAM%2010000%20dataset.png)

For more examples, see the notebook:
> 📓 **`Test_and_Evaulation.ipynb.ipynb`** — contains inference, visualizations, and metric analysis on multiple datasets.

---

## Citation
If you use this work, please consider citing our paper:
> **Skin Disease Segmentation and Harmonizing Global and Local Features via Hyper-Attention Fusion U-Net with SAM Integration**,

---

## License
[MIT License](LICENSE)

---

For any questions or contributions, feel free to open an issue or pull request.
