# Visual Place Recognition (VPR) Toolkit

This toolkit implements, evaluates, and compares VPR methods, with a focus on Bag of Querie (BoQ) method.

## Overview

This repository provides tools for the evaluation of Visual Place Recognition (VPR). The toolkit allows users to:

- Extract features from images using descriptor methods of their choosing
- Evaluate and compare VPR approaches

This repo focuses on comparing a recent BoQ with current techniques such as NetVLAD and CosPlace and extracting features for ORB-SLAM processing.

## Contents

The repository contains the following key notebooks:

### 1. Feature Extraction (`feature_extraction.ipynb`)

This notebook provides a pipeline for extracting visual features from image datasets, with support for the following extraction methods:

- **Holistic descriptors**: AlexNet, SAD
- **Aggregated local descriptors**: NetVLAD
- **Patch-based methods**: PatchNetVLAD
- **Modern methods**: CosPlace, EigenPlaces
- **State-of-the-art**: BoQ-ResNet50, BoQ-DinoV2

Features are saved in `.npy` format for reuse in matching and evaluation tasks on ORB-SLAM.

### 2. VPR Tutorial (`demo.ipynb`)

This notebook offers a walkthrough of a complete VPR pipeline, including:

- Loading image datasets with ground truth correspondence information
- Computing descriptors with various methods
- Creating similarity matrices between image sets
- Performing image matching (single-best and multi-match)
- Calculating evaluation metrics:
  - Precision-Recall curves
  - Recall@K
  - Area Under PR Curve (AUPRC)

### 3. Model Comparison (`model_comparison.ipynb`)

This notebook provides a framework for comparing VPR methods across datasets of interest:

- Evaluate multiple models on standard VPR benchmarks
- Compute 1% recall performance
- Measure and compare processing times
- Generate comparison charts

## Getting Started

### Prerequisites

ROB530ProjectGPU.yaml contains all the necessary packages for the conda environment.

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/Mobile-Robotics-W25-Group-12/VPR_FeatureExtraction.git
   cd VPR_FeatureExtraction
   ```

2. Install required packages:
   ```
   conda env create -f ROB530ProjectGPU.yaml
   conda activate ROB530ProjectGPU
   ```

### Usage

1. **Feature Extraction**:
   - Open `feature_extraction.ipynb`
   - Set the dataset path to your image directory
   - Select your preferred descriptor method
   - Run the notebook to extract and save features

2. **VPR Tutorial**:
   - Open `demo.ipynb`
   - Select a dataset (GardensPoint, StLucia, or SFU)
   - Choose a descriptor method
   - Run the cells to see VPR pipeline in action

3. **Model Comparison**:
   - Open `model_comparison.ipynb`
   - Adjust the list of models and datasets to compare
   - Run the notebook to generate performance comparisons

## Datasets

The toolkit supports the following datasets out-of-the-box (Big thanks to /cite{SchubertVisual}!!):
- GardensPoint Walking
- StLucia (subset)
- SFU Mountain

You can add your own datasets by implementing a dataset loader class similar to those in the `datasets/load_dataset.py` file.

## Supported Methods

The toolkit implements the following VPR methods:

- **AlexNet**: Uses convolutional features from AlexNet
- **NetVLAD**: Neural approximation of VLAD descriptor
- **PatchNetVLAD**: Patch-based extension of NetVLAD
- **HDC-DELF**: Hierarchical Deep Context features with DELF
- **SAD**: Sum of Absolute Differences
- **CosPlace**: Cosine place recognition network
- **EigenPlaces**: Eigenvalue-based place recognition
- **BoQ-ResNet50**: Bag-of-Queries features with ResNet50 backbone
- **BoQ-DinoV2**: Bag-of-Queries features with DinoV2 backbone

## Citation

If you use this toolkit in your research, please cite the following paper:

```bibtex
@article{SchubertVisual,
    title={Visual Place Recognition: A Tutorial},
    author={Stefan Schubert and Peer Neubert and Sourav Garg and Michael Milford and Tobias Fischer},
    journal={arXiv 2303.03281},
    year={2023},
}
```

## Acknowledgments

- Credit to the authors of the implemented methods
- Thanks to the creators of the benchmark datasets 
- Thanks to the UM ROB 530 course instructors, Professor Maani Ghaffari and Minghan Zhu for support.
