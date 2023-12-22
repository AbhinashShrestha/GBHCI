# Gesture Based Human Computer Interaction

### Note: [Python >= 3.11.0 < 3.12](https://www.python.org/downloads/) required 
- Create a python virtual environment
```bash 
python3.11 -m venv major_env
```
- Activate python virtual environment in linux/mac
```bash 
source <env_name>/bin/activate
```
- Activate python virtual environment in windows
```bash 
cd <env_name>/scripts/
activate
``` 
- Deactivate python virtual environment
```bash 
deactivate
```

### Dependencies
To install the required Python packages you can use the following command:
```bash 
pip install opencv-python numpy matplotlib
            Pillow mediapipe
            tensorflow
            PyAutoGUI
            rembg 
            ultralytics supervision
```

for the error OMP: Error #15: Initializing `libomp.dylib`, but found `libiomp5.dylib` already initialized



run the below command before any python program


export `KMP_DUPLICATE_LIB_OK=TRUE`



and if that doesn't work delete `libomp.dylib` file in the env




