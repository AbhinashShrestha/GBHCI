#How to create a python venv named "major_env"

python3.11 -m venv major_env
source major_env/bin/activate

#for deactivating
deactivate



#After the environment is created start installing the following
pip install opencv-python
pip install numpy
pip install matplotlib
pip install Pillow
pip install mediapipe
pip install tensorflow
pip install PyAutoGUI
pip install screen-brightness-control
pip install rembg

for the error OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.

run the following command before executing the program
export KMP_DUPLICATE_LIB_OK=TRUE

deleted libomp.dylib file in the major env
