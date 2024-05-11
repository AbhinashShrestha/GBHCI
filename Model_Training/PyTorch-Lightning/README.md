## Dependencies

Before you begin, ensure you have the following dependencies installed:

- [Python 3.10.0](https://www.python.org/downloads/release/python-3100/)
- [NVIDIA CUDA 12.1.0](https://developer.nvidia.com/cuda-12-1-0-download-archive)

### Environment Setup

Make sure you have the `venv` package installed to create a virtual environment.

- For Windows:
    
    ```bash
    python -m venv env
    source env/scripts/activate
    ```

- For Linux:
    
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

### Install Torch & additional packages

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r requirements.txt
```

## Comet-ML API Integration

For real-time loss curve plotting, edit `.env` with your Comet.ml API key and project name. Click [here](https://www.comet.com/site/) to sign up and get your Comet-ML API key.

```python
API_KEY = "YOUR_API_KEY"
PROJECT_NAME = "YOUR_PROJECT_NAME"
```

### Download Datasets
> _Note: It may take several minutes depending on your internet connection to download and extract_

```bash
python3 dataset_downloader.py
```

### Change Model

If you want to change the architecture of the model, you can choose between `EfficientNet B0 - B7` models and edit in `model.py`.

> _Note: Changing the model version as you go would increase your GPU memory consumption, time, and final model checkpoint file size_

```python 
class neuralnet(pl.LightningModule):
    def __init__(self, num_classes):
        super(neuralnet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1')
        # self.model = EfficientNet.from_pretrained('efficientnet-b2')
        # self.model = EfficientNet.from_pretrained('efficientnet-b5')
```

### Training Model

- On Local Device:

    ```bash
    python3 train.py --train_dir "dataset/asl" --epochs 10 -w 2 --batch_size 128 --precision "16-mixed"
    ```

- On Colab Runtime/Kaggle Kernels:
    >_Note that Colab servers may require additional packages mentioned in `requirements.txt` and uploading of the training scripts_

    ```bash
    !python3 train.py --train_dir "dataset/asl" --epochs 10 -w 2 --batch_size 128 --precision "16-mixed"
    ```

---

Feel free to report any issues you encounter.