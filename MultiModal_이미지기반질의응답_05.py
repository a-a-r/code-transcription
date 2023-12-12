'''
월간 데이콘 이미지 기반 질의 응답 AI 경진대회

알고리즘 | 멀티모달 | 언어 | 비전 | 이미지 기반 질의 응답 | Accuracy
기간 : 2023.07.10 ~ 2023.08.07 10:00 
579명  마감

https://dacon.io/competitions/official/236118/
'''

# [Private 5th] BlipForQuestionAnswering
# https://dacon.io/competitions/official/236118/codeshare/8678?page=1&dtype=recent
# https://github.com/Dongwoo-Im/dacon_vqa

'''
허깅페이스의 BlipForQuestionAnswering 구현체 + official pretrained BLIP weight를 기반으로 VQA task에 fine-tuning 시켰습니다.

전반적으로 가용 메모리가 작아서 (8gb) 메모리 효율적인 학습을 지향하였습니다.
- Freeze image encoder : Locked Image Tuning, BLIP2를 참고하여 freeze
- Gradient checkpointing
- Gradient accumulation (x4)
- Mixed precision training (fp16)
'''


## dataset.py
import os 
from PIL import Image
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(self, df, processor, mode='train'):
        self.df = df
        self.mode = mode 

        self.vis_processor = processor.image_processor 
        self.txt_processor = processor.tokenizer

        if mode == 'test ':
            self.img_path = 'data/image/test'
        else:
            self.img_path = 'data/image/train'

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = os.path.join(self.img_path, row['image_id'] + '.jpg')
        image = Image.open(img_name).convert('RGB')

        inputs = self.txt_processor(
            text=row['question'],
            max_length = 32,
            padding = 'max_length',
            truncation = True,
            return_token_type_ids = False,
            return_tensor = "pt", 
        )
        inputs.update(
            self.vis_processor(
                images=image,
                return_tensor="pt",
            )
        )

        if self.mode != 'test':
            input["labels"] = self.txt_processor(
                text=row['answer'],
                max_length = 32,
                padding='max_length',
                return_tensor = "pt",
            ).input_ids

        for k in inputs.keys():
            inputs[k] = inputs[k].squeeze()

        return inputs


## utils.py
import os
import os.path as osp 
import re 
import time 
import math 
import yaml 
import torch 
import random 
import argparse
import numpy as np

def set_seeds(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    np.random.default_rng(random_seed)

    torch.manual_seedJ(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def get_exp_dir(work_dir):
    work_dir = work_dir.split('./')[-1]
    if not osp.exists(osp.join(os.getcwd(), work_dir)):
        exp_dir = osp.join(os.getcwd(), work_dir, 'exp0')
    else:
        idx = 1
        exp_dir = osp.join(os.getcwd(), work_dir, f'exp{idx}')
        while osp.exists(exp_dir):
            idx += 1
            exp_dir = osp.join(os.getcwd(), work_dir, f'exp{idx}')

    os.makedirs(exp_dir)
    return exp_dir

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False 
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def save_config(args, save_dir):
    with open(save_dir, 'w') as f:
        yaml.safe_dump(args.__dict__, f)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s 
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

## Copied from 
## https://github.com/huggingface/transformers/blob/a865b62e07c88f4139379fea99bace5ac3f400d2/src/transformers/models/blip/convert_blip_original_pytorch_to_hf.py#L54C1-L78C15
def rename_key(key):
    if "visual_encoder" in key: 
        key = re.sub("visual_encoder*", "vision_model.encoder", key)
    if "blocks" in key:
        key = re.sub(r"blocks", "layers", key)
    if "attn" in key:
        key = re.sub(r"attn", "self_attn", key)
    if "norm1" in key:
        key = re.sub(r"norm1", "layer_norm1", key)
    if "norm2" in key:
        key = re.sub(r"norm2", "layer_norm2", key)
    if "encoder.norm" in key:
        key = re.sub(r"encoder.norm", "post_layernorm", key)
    if "encoder.patch_embed.proj" in key:
        key = re.sub(r"encoder.patch_embed.proj", "embeddings.patch_embedding", key)

    if "encoder.pos_embed" in key:
        key = re.sub(r"encoder.pos_embed", "embeddings.position_embedding", key)
    if "encoder.cls_token" in key:
        key = re.sub(r"encoder.cls_token", "embeddings.class_embedding", key)
    if "self_attn" in key:
        key = re.sub(r"self_attn.proj", "self_attn.projection", key)

    return key

## Modified from 
## https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/models/vit.py#L281C1-L305C36
def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.embeddings.num_patches
    num_extra_tokens = visual_encoder.embeddings.num_positions - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchaged
        extra_tokens = pos_embed_checkpoint[:, num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens, dim=1))
        print('reshape position embedding from %d to %d' %(orig_size** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint


## train.py
import time 
import wandb 
import argparse 
import pandas as pd 
import os.path as osp 
from tqdm.auto import tqdm

import torch
import torch.nn as nn 
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from transformers.optimization import get_cosine_schedule_with_warmup 
from transformers.trainer_utils import seed_worker
from transformers import BlipConfig, BlipProcessor, BlipImageProcessor, BertTokenizer, BertTokenizerFast, BlipForQuestionAnswering

from dataset import VQADataset
import utils
try:
    import wandb_utils 
except:
    wandb_utils = None

import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser()
    # Environment 
    parser.add_argument('--work_dir', type=str, default='./work_dirs')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    # Log
    parser.add_argument('--use_wandb', type=utils.str2bool, default=False)
    # Data
    parser.add_argument('--df_ver', type=int, default=1)
    parser.add_argument('--fold', type=int, default=-1)
    # Train
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--grad_accum', type=int, default=4)
    # Model
    parser.add_argument('--pretrained_ckpt', type=str, default='model_base.pth', choices=['model_base.pth', 'model_base_capfilt_large.pth'])
    parser.add_argument('--freeze_image_encoder', type=utils.str2bool, default=True)
    args = parser.parser_args()
    
    return args

def train(epoch, model, loader, optimizer, scheduler, scaler, args, log_freq=1000):
    start = end = time.time()
    batch_time = utils.AverageMeter
    data_time = utils.AverageMeter
    losses = utils.AverageMeter

    model.train()
    optimizer.zero_grad()

    if args.freeze_image_encoder:
        for name, param in model.vision_model.named_parameters():
            param.requires_grad = False

    len_loader = len(loader)
    for i, inputs in enumerate(loader):
        data_time.update(time.time() - end)

        with autocast():
            for k in inputs.keys():
                inputs[k] = inputs[k].to(args.device)

            outputs = model(**inputs)

            loss = outputs.loss / args.grad_accum 
            losses.update(loss.item())
        
        scaler.scale(loss).backward()
        if (i+1) % args.grad_accum == 0 or (i+1) == len_loader:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0* args.grad_accum)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end )
        end = time.time()

        if (i+1) % log_freq == 0 or (i+1) == len_loader:
            print(
                'Epoch {0} [{1}/{2}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Elapsed {remain:s} '
                .format(
                    epoch, i+1, len_loader, 
                    data_time = data_time, 
                    remain = utils.timeSince(start, float(i+1)/len_loader),
                    loss_val = losses.val * args.grad_accum,
                    loss_avg = losses.avg * args.grad_accum,
                )
            )

        if args.use_wandb:
            wandb.log({
                'train_loss' : round(losses.val * args.grad_accum, 4), 
                'learning_rate' : scheduler.optimizer.param_groups[0]['lr'],
            })

    return round(losses.avg * args.grad_accum, 4)

@torch.no_grad()
def evaluation(model, loader, processor, args, device):
    total_bs = 0 
    total_correct = 0
    total_correct_new = 0

    model.eval()

    pbar = tqdm(loader, total=len(loader))
    for inputs in pbar:
        for k in inputs.keys():
            inputs[k] =- inputs[k].to(device)
        outputs = model.generate(**inputs)
        pred = processor.tokenizer.batch_decode(outputs, skip_special_tokens= True)
        gt = processor.tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)
        bs = inputs['labels'].size(0)

        total_bs += bs
        total_correct += sum([g==p for g, p in zip(gt, pred)])

        pbar.set_postfix(
            acc = total_correct/total_bs,
            acc_new = total_correct_new / total_bs
        )

    acc = round(total_correct/total_bs, 4)
    if args.use_wandb:
        wandb.log({'valid_accuracy': acc})
    
    return acc

def main(args):
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # DataFrame
    df = pd.read_csv(f'data/train_5fold_ver{args.df_ver}.csv')
    if args.fold == -1:
        train_df = df
    else:
        train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
        valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)

    # Model
    image_processor = BlipImageProcessor
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    processor = BlipProcessor(image_processor= image_processor, tokenizer=tokenizer)

    model_config = BlipConfig()
    model = BlipForQuestionAnswering(model_config).to(args.device)

    # pretrained weight from (129M & BLIP w/ ViT-B)
    # https://github.com/salesforce/BLIP#pre-trained-checkpoints
    state_dict = torch.laod(args.pretraiend_ckpt)['model']
    for key in state_dict.copy():
        value = state_dict.pop(key)
        if key == 'visual_encoder.pos_embed':
            value = utils.interpolate_pos_embed(value, model.vision_model)
        renamed_key = utils.rename_key(key)
        state_dict[renamed_key] = value
    model.load_state_dict(state_dict, strict=False)

    model.gradient_checkpointing_enable()

    # Dataset & Dataloader
    loader_dict = {"pin_memory": True, "num_workers": 4, "worker_init_fn": seed_worker}
    train_set = VQADataset(train_df, processor, mode='train')
    train_loader = DataLoader(
        train_set, 
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True,
        **loader_dict
    )
    if args.fold != -1:
        if args.freeze_image_encoder:
            valid_batch_size =32
        else:
            valid_batch_size = 16
        valid_set = VQADataset(valid_df, processor, mode='valid')
        valid_loader = DataLoader(valid_set, batch_size=valid_batch_size, **loader_dict)
    
    # Optimizer & Scheduler & GradScaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.num_epochs / args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps= total_steps,
    )

    scaler = GradScaler()

    # Training loop
    valid_acc = 0
    for epoch in range(1, args.num_epochs +1):
        print('-' *10)
        print(f'Epoch {epoch} / {args.num_epochs}')

        train_loss = train(epoch, model, train_loader, optimizer, scheduler, scaler, args)

        if args.fold != -1:
            valid_acc = evaluation(model, valid_loader, processor, args, args.device)
        
        if epoch % args.save_freq == 0:
            file_name = f'epoch{epoch}_acc{valid_acc}.pt'
            torch.save(model.state_dict(), osp.join(args.work_dir_exp, file_name))

        print(f'[Epoch {epoch}] [Train] Loss{train_loss}')
        if args.fold != -1:
            print(f'[Epoch {epoch}] [Valid] Loss{valid_acc}')


if __name__ == "__main":
    args = get_parser()
    args.work_dir_exp = utils.get_exp_dir(args.work_dir)
    args.config_dir = osp.join(args.work_dir_exp, 'config.yaml')
    utils.save_config(args, args.config_dir)
    utils.set_seeds(args.seed)
    if args.use_wandb:
        wandb_utils.wandb_init(args)
    main(args)


## inference.py
import argparse
import pandas as pd 
import os.path as osp 
from tqdm.auto import tqdm 

import torch 
from torch.utils.data import DataLoader

from transformers.trainer_utils import seed_worker
from transformers import Blip2Config, BlipProcessor, BlipImageProcessor, BertTokenizerFast, BlipForQuestionAnswering

from dataset import VQADataset 
import utils 

import warnings 
warnings.filterwarnings("ignore")

@torch.no_grad()
def inference(model, loader, processor, device):
    model.eval()

    preds = []

    for inputs in tqdm(loader, total=len(loader)):
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        outputs = model.generate(**inputs)
        pred = processor.tokenizer.batch_decode(outputs, skip_special_tokens= True)
        
        preds,extend(pred)
    
    return preds

def main(args):
    utils.set_seeds(args.seed)

    args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Data
    test_df = pd.read_csv('data/test.csv')

    # Model
    image_processor = BlipImageProcessor()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    processor = BlipProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model_config = BlipConfig()
    model = BlipForQuestionAnswering(model_config).to(args.device)

    # Load weight
    trained_weight_path = osp.join('work_dirs', args.weight)
    trained_weight = torch.load(trained_weight_path, map_location='cpu')
    model.load_state_dict(trained_weight)

    # Dataset & Dataloader
    loader_dict = {"pin_memory": True, "num_workers": 4, "worker_init_fn": seed_worker}
    test_datset = VQADataset(test_df, processor, mode='test')
    test_loader = DataLoader(test_datset, **loader_dict)

    # Inference
    preds = inference(model, test_loader, processor, args.device)

    # Submission
    submission = pd.read_csv('data/sample_submission.csv')
    submission['answer'] = preds 
    file_path = osp.join('submission', f"{args.weight.replace('/', '_').replace('pt', 'csv')}")
    if osp.exists(file_path):
        file_path = osp.splitext(file_path)[0] + '_dup.csv'
    submission.to_csv(file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight', tyep=str)
    args = parser.parse_args()
    main(args)


## environment.yaml
'''
name: vqa
channels:
  - pytorch
  - defaults
  - conda-forge
dependencies:
  - blas=1.0=mkl
  - brotlipy=0.7.0=py39h2bbff1b_1003
  - ca-certificates=2023.05.30=haa95532_0
  - certifi=2023.7.22=py39haa95532_0
  - cffi=1.15.1=py39h2bbff1b_3
  - charset-normalizer=2.0.4=pyhd3eb1b0_0
  - cryptography=41.0.2=py39hac1b9e3_0
  - cudatoolkit=11.3.1=h59b6b97_2
  - freetype=2.12.1=ha860e81_0
  - giflib=5.2.1=h8cc25b3_3
  - idna=3.4=py39haa95532_0
  - intel-openmp=2023.1.0=h59b6b97_46319
  - jpeg=9e=h2bbff1b_1
  - lerc=3.0=hd77b12b_0
  - libdeflate=1.17=h2bbff1b_0
  - libpng=1.6.39=h8cc25b3_0
  - libtiff=4.5.0=h6c2663c_2
  - libuv=1.44.2=h2bbff1b_0
  - libwebp=1.2.4=hbc33d0d_1
  - libwebp-base=1.2.4=h2bbff1b_1
  - lz4-c=1.9.4=h2bbff1b_0
  - mkl=2023.1.0=h8bd8f75_46356
  - mkl-service=2.4.0=py39h2bbff1b_1
  - mkl_fft=1.3.6=py39hf11a4ad_1
  - mkl_random=1.2.2=py39hf11a4ad_1
  - numpy=1.25.0=py39h055cbcc_0
  - numpy-base=1.25.0=py39h65a83cf_0
  - openssl=3.0.9=h2bbff1b_0
  - pillow=9.4.0=py39hd77b12b_0
  - pip=23.1.2=py39haa95532_0
  - pycparser=2.21=pyhd3eb1b0_0
  - pyopenssl=23.2.0=py39haa95532_0
  - pysocks=1.7.1=py39haa95532_0
  - python=3.9.17=h1aa4202_0
  - pytorch=1.12.1=py3.9_cuda11.3_cudnn8_0
  - pytorch-mutex=1.0=cuda
  - requests=2.29.0=py39haa95532_0
  - setuptools=67.8.0=py39haa95532_0
  - sqlite=3.41.2=h2bbff1b_0
  - tbb=2021.8.0=h59b6b97_0
  - tk=8.6.12=h2bbff1b_0
  - torchaudio=0.12.1=py39_cu113
  - torchvision=0.13.1=py39_cu113
  - typing_extensions=4.6.3=py39haa95532_0
  - urllib3=1.26.16=py39haa95532_0
  - vc=14.2=h21ff451_1
  - vs2015_runtime=14.27.29016=h5e58377_2
  - wheel=0.38.4=py39haa95532_0
  - win_inet_pton=1.1.0=py39haa95532_0
  - xz=5.4.2=h8cc25b3_0
  - zlib=1.2.13=h8cc25b3_0
  - zstd=1.5.5=hd43e919_0
  - pip:
      - albumentations==1.3.1
      - appdirs==1.4.4
      - click==8.1.6
      - colorama==0.4.6
      - docker-pycreds==0.4.0
      - filelock==3.12.2
      - fsspec==2023.6.0
      - gitdb==4.0.10
      - gitpython==3.1.32
      - huggingface-hub==0.16.4
      - imageio==2.31.1
      - joblib==1.3.1
      - lazy-loader==0.3
      - networkx==3.1
      - opencv-python==4.8.0.74
      - opencv-python-headless==4.8.0.74
      - packaging==23.1
      - pandas==2.0.3
      - pathtools==0.1.2
      - protobuf==4.23.4
      - psutil==5.9.5
      - python-dateutil==2.8.2
      - pytz==2023.3
      - pywavelets==1.4.1
      - pyyaml==6.0.1
      - qudida==0.0.4
      - regex==2023.6.3
      - safetensors==0.3.1
      - scikit-image==0.21.0
      - scikit-learn==1.3.0
      - scipy==1.11.1
      - sentry-sdk==1.28.1
      - setproctitle==1.3.2
      - six==1.16.0
      - smmap==5.0.0
      - threadpoolctl==3.2.0
      - tifffile==2023.7.18
      - tokenizers==0.13.3
      - tqdm==4.65.0
      - transformers==4.31.0
      - tzdata==2023.3
      - wandb==0.15.6
prefix: C:\anaconda3\envs\vqa
'''