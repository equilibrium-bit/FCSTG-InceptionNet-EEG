# FCSTG-InceptionNet

This repository contains the PyTorch implementation of **FCSTG-InceptionNet**, a deep learning model for EEG-based diagnostics. The paper has been accepted by **ICASSP 2025**.

## Overview

FCSTG-InceptionNet (Fully Connected Spatio-Temporal Graph InceptionNet) is designed to capture temporal lag and mesoscale spatio-temporal features from EEG signals. The model combines graph neural networks with 3D Inception modules to effectively model the complex spatial and temporal dependencies in brain activity patterns.

### Key Features

- **Temporal Lag Modeling**: Captures temporal dependencies across different time scales using multi-window graph convolution
- **Mesoscale Spatio-Temporal Features**: Extracts features at multiple spatial and temporal resolutions
- **Attention-Based Fusion**: Employs iterative attention mechanisms to fuse multi-scale features
- **3D Inception Architecture**: Processes spatially-mapped EEG data with multi-scale convolutional kernels

## Model Architecture

The model consists of several key components:

1. **Feature Extraction**: 1D CNN extracts temporal features from raw EEG signals
2. **Graph Construction**: Dynamic graph construction based on electrode relationships
3. **Multi-Window MPNN**: Message passing neural networks with different temporal windows to capture multi-scale patterns
4. **Electrode Space Mapping**: Maps EEG channels to 3D spatial coordinates based on electrode positions
5. **Attention Fusion**: Iterative attention mechanism fuses features from different temporal scales
6. **3D Inception Classifier**: Final classification using 3D Inception modules

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- CUDA (recommended for GPU acceleration)

### Setup

```bash
git clone https://github.com/yourusername/FCSTG-InceptionNet.git
cd FCSTG-InceptionNet
pip install torch numpy
