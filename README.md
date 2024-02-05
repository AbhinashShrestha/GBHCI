# Gesture Based Human-Computer Interaction

This is an ongoing project that enables interaction with computers using hand gestures, leveraging advanced deeplearning and computer vision techniques.


## Prerequisites

Ensure you have Python version 3.11.0 or later, but not version 3.12 or later. You can download the appropriate version of Python from the official website.
You should have the model required for the project too.

## Setting Up the Environment

1. **Create a Python virtual environment:**

    ```bash 
    python3.11 -m venv major_env
    ```
    windows powershell 
    Install The Powershell Extension in VSCode
    Then open the Powershell Extension terminal
    Open Command Prompt 
    ```
    where python
    ```
    Now copy that path and 
    ```
    C:\Users\username\AppData\Local\Programs\Python\Python311\python.exe -m venv major_env
    ```
    
    ```
    .\\major_env\Scripts\Activate.ps1
    ```
2. **Activate the Python virtual environment:**
    - On Linux/Mac:
        ```bash 
        source major_env/bin/activate
        ```
    - On Windows:
    Use the powershell extension terminal
        ```
        major_env\Scripts\Activate.ps1
        ```
3. **Deactivate the Python virtual environment when done:**
    ```bash 
    deactivate
    ```

## Installing Dependencies

Install the required Python packages using the following command:

```bash 
pip install opencv-python numpy matplotlib Pillow mediapipe tensorflow PyAutoGUI rembg ultralytics supervision
```

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
