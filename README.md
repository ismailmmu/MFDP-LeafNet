# MFDP-LeafNet


**MFDP-LeafNet: A Few-Shot Learning Method for Plant Species Classification Using Multi-Feature and Historical Dynamic Prototypical Network**

Published in *Multimedia Tools and Applications*, 2026.

**DOI:** https://doi.org/10.1007/s11042-026-21554-6

---

## Overview

MFDP-LeafNet is a few-shot learning framework designed for plant species classification under limited labelled data conditions. The proposed approach integrates complementary representations extracted from multiple deep convolutional neural networks and employs a Historical Dynamic Prototypical Network (HDPN) to improve class prototype estimation in few-shot scenarios.

The framework consists of:

- Multi-Backbone Feature Integration (MFI)
  - ResNet152
  - DenseNet121
  - MobileNetV2
- Sparse Autoencoder-Based Feature Compression
- Historical Dynamic Prototypical Network (HDPN)
- Episodic Few-Shot Classification using EasyFSL
- Grad-CAM Visualisation for Model Interpretability

The proposed method was evaluated on plant leaf classification datasets and demonstrated superior performance compared with several state-of-the-art few-shot learning approaches.

---

## Repository Structure

```text
MFDP-LeafNet/
│
├── README.md
├── requirements.txt
├── main.py
├── model.py
├── feature_extractor.py
├── feature_layer_gradient.py
├── helper.py
│
├── sparse_autoencoder/
│   ├── autoencoder.py
│   └── train.py
│
├── weights/
│   └── sparse_auto_encoder2500.pth
│
└── figures/
    └── architecture.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ismailmmu/MFDP-LeafNet.git
cd MFDP-LeafNet
```

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

The code uses PyTorch's `ImageFolder` format. Organise the dataset as follows:

```text
dataset_root/
├── Class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Class_2/
│   ├── image1.jpg
│   └── ...
└── ...
```

Update the dataset path in `main.py`:

```python
val_dir = "path_to_dataset/EgyptianPlantLeaf"
```

---

## Training the Sparse Autoencoder

MFDP-LeafNet employs a sparse autoencoder to learn compact feature representations before prototype generation.

### Step 1: Configure Dataset Path

Open:

```text
sparse_autoencoder/train.py
```

Update the dataset path according to your local environment.

### Step 2: Train the Autoencoder

Run:

```bash
python sparse_autoencoder/train.py
```

After training, the generated checkpoint should be saved in:

```text
weights/
└── sparse_auto_encoder2500.pth
```

---

## Running MFDP-LeafNet

Before running the experiment, verify that the autoencoder checkpoint path in `main.py` is correct:

```python
weight_path = "weights/sparse_auto_encoder2500.pth"
```

Execute:

```bash
python main.py
```

---



To reproduce the experiments reported in the paper:

1. Prepare the dataset using the ImageFolder format.
2. Train the sparse autoencoder.
3. Save the trained checkpoint inside the `weights/` directory.
4. Update dataset and weight paths.
5. Run:

```bash
python main.py
```

6. Compare the generated performance metrics with those reported in the paper.

---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{hossen2026mfdpleafnet,
  title={MFDP-LeafNet: A few-shot learning method for plant species classification using multi-feature and historical dynamic prototypical network},
  author={Hossen, Md Ismail and Awrangjeb, Mohammad and Pan, Shirui and Islam, Mohammad Aminul and Al Mamun, Abdullah},
  journal={Multimedia Tools and Applications},
  year={2026},
  doi={10.1007/s11042-026-21554-6}
}
```

---

## Notes

- Large datasets are not included in this repository.
- Trained model weights are not distributed by default.
- Users may need to update dataset paths according to their local environment.


---

## License

This repository is provided for academic and research purposes. If you use this code or any part of the methodology in your research, please cite the associated publication.