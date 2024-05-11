import torch
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model import neuralnet
from dataset import HandSignDataModule
import argparse

def main(args):
    # Setting up device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = HandSignDataModule(train_dir=args.train_dir, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.data_workers)
    
    # Call setup to initialize datasets
    dataloader.setup('fit')  
    num_classes = dataloader.get_num_classes()

    # Initialize the model
    model = neuralnet(num_classes=num_classes).to(device)

    # Create a checkpoint callback
    # Save the model periodically by monitoring a quantity
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",                    # Quantity to monitor | By default: None(saves last chekpoint)
                                          dirpath="saved_checkpoint/",           # Directory to save the model file.
                                          filename="model-{epoch:02d}-{val_loss:.2f}")   # Checkpoint filename

    # Create a Trainer instance for managing the training process.
    trainer = pl.Trainer(accelerator=device,
                         devices=args.gpus,
                         min_epochs=1,
                         max_epochs=args.epochs,
                         precision=args.precision,
                         callbacks=[EarlyStopping(monitor="val_loss"),   # Monitor a metric and stop training when it stops improving
                                    checkpoint_callback])                # Pass defined checkpoint callback

    # Fit the model 
    trainer.fit(model, dataloader)
    trainer.validate(model, dataloader)
    # trainer.test(model, dataloader)


if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Train")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--data_workers', default=0, type=int,
                        help='n data loading workers, default 0 = main process only')

    # Train Directory Params
    parser.add_argument('--train_dir', default=None, required=True, type=str,
                        help='Folder path to load training data')
    
    # General Train Hyperparameters
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('--precision', default=16, type=str, help='precision')
    
    args = parser.parse_args()
    main(args)