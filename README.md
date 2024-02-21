"""
# Gesture-Based Human-Computer Interaction

This project is a work in progress that allows for interaction with computers using hand gestures. It utilizes machine learning and computer vision techniques.

## Prerequisites

Please ensure you have Python version 3.11.0 or later installed, but not version 3.12 or later. You can download the appropriate version of Python from the official website. Additionally, you should have the model required for this project.

## Environment Setup

1. Create a Conda environment:
    # conda env create -f environment.yml

2. Activate the Python virtual environment:
    # conda activate major

3. Deactivate the Python virtual environment when done:
    # conda deactivate

## Dependency Installation

Install CUDA (required only for training). This requires Python 3.10 and TensorFlow 2.10. Note that Google does not support Windows natively and my WSL does not open.
    # conda create -n major_tensorflow python=3.10
    # conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    # conda install -c nvidia cuda-nvcc

To test if your GPU is detected, run the following command:
    # python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

## Usage

To use the application, run the following command:
```bash
python3 gesture_efficientnet.py
```

## Contributions

All contributions are welcome and encouraged.

## Troubleshooting

If you encounter the error 'OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized', run the following command before running any Python program:
    # export KMP_DUPLICATE_LIB_OK=TRUE

If the error persists, delete the 'libomp.dylib' file in the environment.
"""

If you need further help. Open an Issue.
