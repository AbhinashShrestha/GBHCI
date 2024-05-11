import os
from dotenv import load_dotenv
from comet_ml import Experiment

import torch
from torch import nn 
from efficientnet_pytorch import EfficientNet
import pytorch_lightning as pl 
import torchmetrics

# Load environment variables from .env file
load_dotenv()
API_KEY, PROJECT_NAME = os.getenv("API_KEY", "PROJECT_NAME")

# Initalize model and training global steps
class neuralnet(pl.LightningModule):
    def __init__(self, num_classes):
        super(neuralnet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        # Modify the classifier layer
        # Get the number of input features to the classifier
        num_ftrs = self.model._fc.in_features  
        self.model._fc = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True),
                                             torch.nn.Linear(in_features=num_ftrs, 
                                                             out_features=num_classes, 
                                                             bias=True))
        
        # Setup Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes) 
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes) 
        self.experiment = Experiment(api_key=API_KEY, project_name=PROJECT_NAME)
        
    def forward(self, x):
        return self.model(x)
    
    def _common_step(self, batch, batch_idx):
        X, y = batch                       
        y_pred = self.forward(X)       
        loss = self.loss_fn(y_pred, y) 
        return loss, y_pred, y
    
    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)  
        f1_score = self.f1_score(y_pred, y)
        logs = {"loss": loss, "accuracy": accuracy, "f1_score": f1_score}
        self.log_dict(logs, on_step=True, on_epoch=False, 
                            prog_bar=True, logger=True)
        self.experiment.log_metric('train_loss', loss.item(), step=self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx) 
        accuracy = self.accuracy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)        # Log valdiation loss in prog bar
        self.log('val_acc', accuracy, prog_bar=True)     # Log accuracy loss in prog bar
        self.experiment.log_metric('val_loss', loss.item(), step=self.global_step)
        return loss
    
    def test_step(self, batch, batch_idx):
        pass
    
    def predict_step(self, batch, batch_idx):
        X = batch
        y_pred = self.forward(X)
        preds = torch.argmax(y_pred, dim=1) 
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    