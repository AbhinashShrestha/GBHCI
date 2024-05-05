# Gesture-Based Human-Computer Interaction

This project enables interaction with computers using hand gestures by leveraging machine learning and computer vision techniques.

## Table of Contents
1. Prerequisites
2. Environment Setup
3. Dependency Installation
4. Usage
5. Contributions
6. License and Support

## Prerequisites

Please ensure you have Python version 3.11.0 or later installed, but not version 3.12 or later. You can download the appropriate version of Python from the official website. Additionally, you should have the model required for this project.

## Environment Setup

1. Create a Conda environment:
```bash
conda env create -f environment.yml
```
2. Activate the Python virtual environment:
```bash
conda activate major
```
3. Deactivate the Python virtual environment when done:
```bash
conda deactivate
```

## Dependency Installation

Install CUDA (required only for training). This requires Python 3.10 and TensorFlow 2.10. Note that Tensorflow does not support Windows natively after version 2.10. All training must be done in WSL2:

```bash
conda create -n major_tensorflow python=3.10
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c nvidia cuda-nvcc
```

To test if your GPU is detected, run the following command:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Usage

To use the application, run the following command:
```bash
python3 gesture_efficientnet.py
```

## Contributions

CLOSED UNTIL FURTHER NOTICE

## License and Support

This project is licensed under the Apache License. You are free to use, modify, and distribute the code as per the terms of the license.

While the code is open-source, please note that any support or assistance provided in relation to this project is not free. If you require support or assistance in using this project, please contact me for details about support options and pricing.
