# Gesture-Based Human-Computer Interaction
This project is a work in progress that allows for interaction with computers using **hand gestures**. It utilizes **machine learning** and **computer vision** techniques.

## Prerequisites
Please ensure you have [`Python version >= 3.11.0`](https://www.python.org/downloads/release/python-3110/) or later installed, but `< 3.12`. Additionally, you should have the model required for this project.

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
Install **CUDA** (required only for training). This requires `Python 3.10` and `TensorFlow 2.10`.
*Note: Google does not support Windows natively and WSL does not open.*
```bash
conda create -n major_tensorflow python=3.10
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c nvidia cuda-nvcc
```

To test if your **GPU** is detected, run the following command:

```bash
python -c "import tensorflow as tf; 
            print(tf.config.list_physical_devices('GPU'))"
```

## Usage

To use the application, run the following command:
```bash
python3 gesture_efficientnet.py
```

## Troubleshooting

If you encounter the error `OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized`. Run the following command before running any Python program:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```
*If the error persists, delete the `libomp.dylib` file in the environment.*

## Contributors <img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="25px" />
<a href="https://github.com/AbhinashShrestha/GBHCI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AbhinashShrestha/GBHCI"/>
</a>
</br>

---
_Feel free to customize and extend the **Gesture-Based Human-Computer Interaction** to suit your specific needs and requirements.</br>
Contributions and feedback are welcome!</br>
If you need further help. Open an Issue._
