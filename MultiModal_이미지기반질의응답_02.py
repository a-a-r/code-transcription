'''
월간 데이콘 이미지 기반 질의 응답 AI 경진대회

알고리즘 | 멀티모달 | 언어 | 비전 | 이미지 기반 질의 응답 | Accuracy
기간 : 2023.07.10 ~ 2023.08.07 10:00 
579명  마감

https://dacon.io/competitions/official/236118/
'''

# [Private 3위, 0.62888] BLIP hard voting
## https://dacon.io/competitions/official/236118/codeshare/8679?page=1&dtype=recent

## BLIP
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

import transformers
from transformers import AutoProcessor
from transformers import BlipForQuestionAnswering
transformers.logging.set_verbosity_error()

import re
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser

# import wandb
# from pytorch_lightning.loggers import WandbLogger
# wandb_logger = WandbLogger(name='BLIP', project='Dacon_236118')

parser = ArgumentParser(description='BLIP')
parser.add_argument('--vision_language_pretrained_model', default='blip-itm-base-coco', type=str)
parser.add_argument('--image_size', default=384, type=int)
parser.add_argument('--text_len', default=32, type=int)
parser.add_argument('--optimizer', default='adamw', type=str)
parser.add_argument('--learning_rate', default=0.00001, type=float)
parser.add_argument('--scheduler', default='none', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--validation_size', default=0.1, type=int)
parser.add_argument('--seed', default=826, type=int)
parser.add_argument('--mixed_precision', default=16, type=int)
parser.add_argument('--device', default=[0], type=list)
parser.add_argument('--num_workers', default=0, type=int)
args = parser.parse_args('')

# wandb.config.update(args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

image_size = args.image_size
text_len = args.text_len
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
VALIDATION_SIZE = args.validation_size
SEED = args.seed

def set_seeds(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(SEED)

set_seeds()

idx = f'{args.vision_language_pretrained_model}_{args.seed}'
print(idx)


## run.py
if args.vision_language_pretrained_model == 'blip-itm-base-coco':
    vl_model_name = 'Salesforce/blip-itm-base-coco'

processor = AutoProcessor.from_pretrained(vl_model_name)
vl_model = BlipForQuestionAnswering.from_pretrained(vl_model_name)

vocap_size = len(processor.tokenizer)
print(vocap_size)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

train_img_path = 'data/image/train'
test_img_path = 'data/image/test'

train_df.head()

train_df = train_df.drop_duplicates(
    subset=['image_id', 'question', 'answer']
).reset_index(drop=True)
train_df.shape


## data_loader.py
class VQADataset(Dataset):
    def __init__(self, df, img_path, is_test=False):
        self.df = df
        self.processor = processor
        self.img_path = img_path
        self.is_test = is_test

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(self.img_path + row['image_id'] + '.jpg').convert('RGB')
        
        question = row['question']
        
        encoding = self.processor(
            image, question,
            add_special_tokens = True,
            padding = 'max_length',
            truncation = True,
            max_length = text_len,
            return_tensors = 'pt'
        )

        if not self.is_test:
            answer = row['answer']

            targets = self.processor.tokenizer(
                answer,
                add_special_token = True,
                padding = 'max_lenght',
                truncation = True,
                max_length = text_len,
                return_tensors = 'pt'
            )

            encoding['labels'] = targets['input_ids']

        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        
        return encoding
    

## model.py
class VQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vl_model = vl_model

    def forward(self, inputs):
        outputs = self.vl_model(**inputs)
        return outputs
    
class VQAClassifier(pl.LightningModule):
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch):
        predictions = self.backbone(batch)
        return predictions
    
    def step(self, batch):
        y = batch["labels"]
        outputs = self.backbone(batch)
        loss = outputs.loss
        y_hat = self.backbone.vl_model.generate(**batch)
        return loss, y, y_hat
    
    def compute_accuracy(self, y_hat, y):
        preds = []
        for i in range(len(y)):
            pred = processor.decode(y_hat[i], skip_special_tokens=True)
            true = processor.decode(y[i], skip_special_tokens=True)
            preds.append(pred==true)
        acc = np.sum(preds) / len(preds)
        return acc
    
    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        acc = self.compute_accuracy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        acc = self.compute_accuracy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        acc = self.compute_accuracy(y_hat, y)
        self.log('test_acc', acc)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self.backbone.v1_model.generate(**batch)
        return y_hat
    
    def configure_optimizers(self):
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=args.learning_rate, momentum=0.9)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=args.learning_rate)

        if args.scheduler == 'none':
            return optimizer
        if args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size = 2,
                gamma=0.9,
            )
            return [optimizer], [scheduler]
        if args.scheduler == 'onecyclelr':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer = optimizer,
                max_lr=args.learning_rate,
                epochs=EPOCHS,
                steps_per_epoch=int(len(train_index) / BATCH_SIZE),
                pct_start=0.1,
            )
            return [optimizer], [scheduler]


## main.py
### preprocessing.py

temp_df, val_df = train_test_split(train_df, test_size=VALIDATION_SIZE, random_state=SEED)

temp_df = temp_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

### data_loaders.py

train_ds = VQADataset(temp_df, train_img_path, is_test=False)
val_ds = VQADataset(val_df, train_img_path, is_test=False)
test_ds = VQADataset(test_df, test_img_path, is_test=True)

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)

### train.py

model = VQAClassifier(VQAModel(), args)

callbacks = [
    pl.callbacks.ModelCheckpoint(
        dirpath='saved/', filename=f"{idx}",
        monitor='val_acc', mode='max',
    )
]

trainer = pl.Trainer(
    max_epochs=EPOCHS, accelerator='auto', callbacks=callbacks,
    precision=args.mixed_precision, # logger=wandb_logger,
    device=args.device,
)

trainer.fit(model, train_dataloader, val_dataloader)

ckpt = torch.load(f'saved/{idx}.ckpt', map_location=torch.device(device))
model.load_state_dict(ckpt['state_dict'])

### test.py

eval_dict = trainer.validate(model, dataloaders=val_dataloader)[0]
# wandb.log({'val_accuracy': eval_dict['val_acc']})

y_preds = trainer.predict(model, dataloaders=test_dataloader)


## Submission
outputs = []
for y_pred in y_preds:
    for output in y_pred:
        outputs.append(processor.decode(output, skip_special_tokens=True))

sample_submission['answer'] = outputs
sample_submission.to_csv(f'{idx}.csv', index=False)

# wandb.finish()