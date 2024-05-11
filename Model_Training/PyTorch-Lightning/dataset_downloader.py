""" Download and Extract dataset from google drive"""

import gdown
import rarfile
import os

file_url = "https://drive.google.com/uc?id=1b0-MLad_AcVvbocCk7RUB2XH5Xbr7L3x"
file_name = "ASL.rar"

# Download the file from Google Drive and extract
gdown.download(file_url, file_name, quiet=False)
extract_dir = './dataset'

# Extract the RAR file
with rarfile.RarFile(file_name, 'r') as rar_ref:
    rar_ref.extractall(extract_dir)

# Remove the RAR file after extraction
os.remove(file_name)

print("Files extracted successfully to:", extract_dir)