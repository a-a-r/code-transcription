'''
월간 데이콘 이미지 기반 질의 응답 AI 경진대회

알고리즘 | 멀티모달 | 언어 | 비전 | 이미지 기반 질의 응답 | Accuracy
상금 : 인증서
기간 : 2023.07.10 ~ 2023.08.07 10:00 
579명  마감

https://dacon.io/competitions/official/236118/leaderboard
'''

# Private 1위, BEiT-3 large 모델, (DistributedDataParallel 활용 가능)
## https://github.com/HwangGyuYoung/dacon_vqa/blob/main/dataset.py
## dataset

import json
import os
import torch
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torch.utils.data import Dataset

from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image

from beit3.randaug import RandomAugment

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

def build_transform(is_train, img_size):
    if is_train:
        t = [
            RandomResizedCropAndInterpolation(img_size, scale=(0.5, 1.0), interpolation='bicubic'),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True,
                          augs=[
                              'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'
                          ]
            )
        ]

    else:
        t = [
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC)
        ]

    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ]
    t = transforms.Compose(t)

    return t


class VQADataset(Dataset):
    def __init__(self, df, tokenizer, img_path, *, img_size=480, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = build_transform(is_train, img_size)
        self.img_path = img_path
        self.is_train = is_train

        ans2label_file = os.path.join(base_path, "answer2label.txt")
        ans2label = {}
        label2ans = []
        with open(ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                ans = data["answer"]
                label = data["label"]
                label = int(label)
                assert label == 1
                ans2label[ans] = torch.tensor(i)
                label2ans.append(ans)

        self.ans2label = ans2label
        self.label2ans = label2ans

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = os.path.join(base_path, self.img_path, row['image_id' + '.jpg'])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        question = row['question']
        question = self.tokenizer.encode_plus(
            question,
            truncation = True,
            add_special_tokens = True,
            max_length = 32, 
            padding = 'max_length',
            return_attention_mask = True,
            return_tensor = 'pt'
        )

        if self.is_train:
            answer = row['answer']
            try:
                label = self.ans2label[answer]
                one_hots = torch.nn.functional.one_hot(label, num_classes=3129)
            except KeyError:    # 3129개 이외 클래스에 해당되는 답변 예외 처리
                one_hots = torch.tensor([0]*3129)

            return {
                'image' : image.squeeze(),
                'question' : question['input_ids'].squeeze(),
                'padding_mask' : question['attention_mask'].squeeze().logical_not().to(int),
                'answer' : one_hots.squeeze()
            }
        
        else:
            return {
                'image' : image,
                'question' : question['input_ids'].squeeze(),
                'padding_mask' : question['attention_mask'].squeeze().logical_not().to(int)
            }
        

## https://github.com/HwangGyuYoung/dacon_vqa/blob/main/distributed_train.py
## distributed_train

from dataclasses import dataclass
import datetime
import os
import random
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.dstributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import XLMRobertaTokenizer, get_cosine_schedule_with_warmup
from timm.models import create_model

from beit3 import utils, modeling_finetune
from dataset import VQADataset, base_path

import warnings
warnings.filterwarnings(action='ignore')

### Hyperparameter Setting
CFG = {
    'MODEL_SIZE' : 'large',
    'IMAGE_SIZE' : 480,
    'EPOCHS' : 10,
    'LEARNING_RATE' : le-5,
    'BATCH_SIZE' : 8,
    'SEED' :  41
}

@dataclass
class DistributedArgs:
    world_size: int = 4
    gpu: tuple = (0, 1, 2, 3)
    dist_url: str = 'tcp://0.0.0.0:37860'
    dist_backend: str = 'nccl'

# Fixed RandomSeed
random.seed(CFG['SEED'])
os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_startegy('file_system')


def main():
    args = DistributedArgs()
    mp.spqwn(main_worker, args=(args,), nprocs=args.world_size, join=True)

def main_worker(rnak, args):
    torch.cuda.set_device(args.gpu[rank])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dist.init_process_group(
        backend = args.dist_backend, init_method =args.dist_url,
        world_size = args.world_size, rank = rank,
        timeout = datetime.timedelta(0, 7200)
    )
    torch.distributed.barrier()

    # Data Load
    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    train_img_path = 'image/train'

    # Dataset & Dataloader
    tokenizer = XLMRobertaTokenizer(os.path.join(base_path, 'models', 'beit3.spm'))
    train_dataset = VQADataset(train_df, tokenizer, train_img_path, img_size=CFG['IMAGE_SIZE'], is_train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=4, sampler=train_sampler, pin_memory=True)

    # Model Load
    model_config = f'beit3_{CFG["MODEL_SIZE"]}_patch16_{CFG["IMG_SIZE"]}_vqav2'
    model = create_model(
        model_config,
        pretrained = False,
        drop_path_rate = 0.4,
        vocab_size = 64010
    )

    utils.load_model_and_may_interpolate(
        ckpt_path = os.path.join(base_path, 'models', f'beit3_{CFG["MODEL_SIZE"]}_indomain_patch16_224.zip'),
        model=model,
        model_key='model|module',
        model_prefix=''
    )
    model.to(device)
    model = DDP(model, device_ids=[args.gpu[rank]],find_unused_parameters=True)

    # Train
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=CFG["LEARNING_RATE"], betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer, 
        num_warmup_steps = len(train_loader) * int(CFG["EPOCHS"] * 0.1),
        num_training_steps = len(train_loader) * CFG["EPOCHS"]
    )
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(1, CFG['EPOCHS']+1):
        total_loss = 0

        for data in tqdm(train_loader, total=len(train_loader)):
            images = data['image'].to(device)
            question = data['question'].to(device)
            padding_mask = data['padding_mask'].to(device)
            answer = data['answer'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images, question, padding_mask)
                loss = criterion(input=outputs.float(), target=answer.float())
            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.updata()

            scheduler.step()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch}/{CFG["EPOCHS"]}], Train Loss: [{avg_loss:.5f}]')

            torch.save(
                model.state_dict(),
                os.path.join(
                    base_path, 'models',
                    f'{epoch}_{CFG["EPOCHS"]}_{"{:.0e}".format(CFG["LEARNING_RATE"])}_large_model.pt'
                )
            )


if __name__ == '__main__':
    main()


## https://github.com/HwangGyuYoung/dacon_vqa/blob/main/make_answer2label.py
## make_answer2label

from collections import Counter
import json
import os

import pandas as pd

from dataset import base_path

train_df = pd.read_csv(os.pth.join(base_path, 'train.csv'))
counter = Counter(train_df['answer'])
sorted_dict = sorted(counter.items(), key=lambda item: item[1], reverse=True)

with open('answer2label.txt', mode='w', encoding='utf-8') as writer:
    for i, (k, _) in enumerate(sorted_dict[:3129]):
        to_json = {
            "answer" : k,
            "label" : i
        }
        writer.write("%S\n" % json.dumps(to_json))


## https://github.com/HwangGyuYoung/dacon_vqa/blob/main/train.py
## train

import os
import random
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data import DataLoader
from transformer import XLMRobertaTokenizer, get_cosine_schedule_with_warmup
from timm.models import create_model
from beit3 import utils, modeling_finetune
from dataset import VQADataset, base_path
import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

### Hyperparameter Setting
CFG = {
    'IMAGE_SIZE': 480,
    'EPOCHS': 10, 
    'LEARNING_RATE': 1e-5, 
    'BATCH_SIZE': 16, 
    'SEED': 41, 
}

### Fixed RandomSeed
random.seed(CFG['SEED'])
os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])
torch.backends.cudnn.deterministric = True
torch.backends.cudnn.benchmark = True

### Data Load
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
train_img_path = 'image/train'

### dataset & dataloader
tokenizer = XLMRobertaTokenizer(os.path.join(base_path, 'models', 'beit3.spm'))
train_dataset = VQADataset(train_df, tokenizer, train_img_path, img_size=CFG['IMG_SIZE'], is_train=True)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=8)

### Model Load
model_config = 'beit3_large_patch16_480_vqav2'
model = create_model(
    model_config,
    pretrained=False,
    drop_path_rate=0.4,
    vocab_size=64010
)

utils.load_model_and_may_interpolate(
    ckpt_path = os.path.join(base_path, 'models', 'beit3_large_indomain_patch16_224.zip'),
    model = model,
    model_key = 'model|module',
    model_prefix=''
)
model = torch.comile(model)

### Train
criterion = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.AdamW(params=model.parameters(), lr=CFG['LEARNING_RATE'], betas=(0.9, 0.999), weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer = optimizer,
    num_warmup_steps = len(train_loader) * int(CFG["EPOCHS"]*0.1),
    num_training_steps = len(train_loader) * CFG["EPOCHS"]
)

model.train()
model.to(device)
for epoch in range(1, CFG['EPOCHS']+1):
    total_loss = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        images = data['image'].to(device)
        question = data['question'].to(device)
        padding_mask = data['padding_mask'].to(device)
        answer = data['answer'].to(device)

        optimizer.zero_grad()

        outputs = model(images, question, padding_mask)

        loss = criterion(input=outputs.float(), target=answer.float())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}/{CFG['EPOCHS']}], Train Loss: [{avg_loss:.5f}]")

    torch.save(
        model.state_dict(),
        os.path.join(base_path, 'models', f'{epoch}_{CFG["EPOCHS"]}_{"{:.0e}".foramt(CFG["LEARNING_RATE"])}_large_model.pt')
    )


## https://github.com/HwangGyuYoung/dacon_vqa/blob/main/inference.py
## inference

import os
from collections import OrderedDict
import pandas as pd
import torch
from torch.utils.data imoprt DataLoader
from tqdm.auto import tqdm
from timm.models import create_model
from transformers import XLMRobertaTokenizer
from beit3 import modeling_finetune
from dataset import VQADataset, base_path

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

### Hyperparameter Setting
CFG = {
    'MODEL_SIZE': 'large', 
    'IMG_SIZE': 480
}

### Data Load
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
test_img_path = 'image/test'

### Dataset & DataLoader
tokenizer = XLMRobertaTokenizer(os.path.join(base_path, 'models', 'beit3.spm'))
test_datset = VQADataset(test_df, tokenizer, test_img_path, img_size=CFG['IMG_SIZE'], is_train= False)
test_loader = DataLoader(test_datset, batch_size=64, shuffle=False, num_workers=8)

### Model Load
model_config = f'beit3_{CFG["MODEL_SIZE"]}_patch16_{CFG["IMG_SIZE"]}_vqav2'
model = create_model(
    model_config,
    pretrained = False,
    drop_path_rate = 0.1,
    vocab_size = 64010
)

tmp_weight = torch.load(os.path.join(base_path, 'models', '7_10_1e-05_large_model.pt'))
weight = OrderedDict()
for k, v in tmp_weight.items():
    weight[k[10:]] = v          # train.py 결과물 -> 10, distributed_train.py 결과물 -> 7

model.load_state_dict(weight)
model.eval()
model.to(device)

preds = []
with torch.no_grad():
    for data in tqdm(test_loader, total=len(test_loader)):
        images = data['image'].to(device)
        question = data['question'].to(device)
        padding_mask = data['padding_mask'].to(device)

        outputs = model(images, question, padding_mask)

        _, pred = outputs.max(-1)
        for x in pred:
            preds.append(test_datset.label2ans[x])

sample_submission['answer'] = preds
sample_submission.to_csv('submission.csv', index=False)