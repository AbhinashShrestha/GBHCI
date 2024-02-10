# Gesture Based Human-Computer Interaction

This is an ongoing project that enables interaction with computers using hand gestures, leveraging advanced deeplearning and computer vision techniques.


## Prerequisites

Ensure you have Python version 3.11.0 or later, but not version 3.12 or later. You can download the appropriate version of Python from the official website.
You should have the model required for the project too.

## Setting Up the Environment

1. **Create a Conda environment:**

    conda create -n major python=3.11.5

2. **Activate the Python virtual environment:**
    - On Linux/Mac:
        ```bash 
        conda activate major
        ```
3. **Deactivate the Python virtual environment when done:**
    ```bash 
    deactivate
    ```

## Installing Dependencies
Install Cuda
Install the required Python packages using the following command:
``` bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
tensorflow route
requires python 3.10 and tensorflow 2.10
Cuz google is not supporting windows native and my wsl doesnt open
```
conda create -n major_tensorflow python=3.10
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c nvidia cuda-nvcc
```
test if your gpu is detected 
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```


install 
conda install conda-forge::opencv
pip install mediapipe
pip install pydot
## How to use

```bash
 python3 gesture.py
```
## Contribution

All contribution are welcome and encouraged.



## Troubleshooting
If you encounter the error OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized, run the following command before running any Python program:

export KMP_DUPLICATE_LIB_OK=TRUE

If the error persists, delete the libomp.dylib file in the environment.
