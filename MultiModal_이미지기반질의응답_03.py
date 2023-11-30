'''
월간 데이콘 이미지 기반 질의 응답 AI 경진대회

알고리즘 | 멀티모달 | 언어 | 비전 | 이미지 기반 질의 응답 | Accuracy
기간 : 2023.07.10 ~ 2023.08.07 10:00 
579명  마감

https://dacon.io/competitions/official/236118/
'''

# [Public 3rd, Private 2nd] LLaVA
# https://dacon.io/competitions/official/236118/codeshare/8669?page=1&dtype=recent
# https://github.com/geuk-hub/-Dacon-Multimodal-vqa
# LLaVA 모델 구현은 https://llava-vl.github.io 를 참고하였습니다.

## 1. Introduction
'''
[설명] 이미지의 id, 해당 이미지와 관련된 질문이 담긴 csv 파일이 실제 이미지와 함께 데이터셋으로 제공. 
이미지에서 확인할 수 있는 정보들을 바탕으로, 질문에 대해 올바르게 답변할 수 있는 AI 모델을 개발하는 것이 목표.
'''
## 2. Data
'''
data
├─  image
│   ├─  train : 107,231개
│   │   ├─  train_000000.png
│   │   ├─  train_000001.png
│   │   └─  ...
│   └─  test : 11,915개
│       ├─  test_00000.png
│       ├─  test_00001.png
│       └─  ...
├─  train.csv
|    ├─  ID : 질문 ID
|    ├─  image_id : 이미지 ID
|    ├─  question : 이미지 관련 질문
|    └─  answer : 질문에 대한 답변
├─  test.csv
|    ├─  ID : 질문 ID
|    ├─  image_id : 이미지 ID
|    └─  question : 이미지 관련 질문
└─  sample_submission.csv
     ├─  ID : 질문 ID
     └─  *answer : 질문에 대한 답변
'''

## 3. Setup
'''
- Colab-Pro or Pro+ 
- GPU A100
'''

### Clone LLaVa
!git clone https://github.com/haotian-liu/LLaVA.git
%cd /content/LLaVA

### Install
!pip install --upgrade pip
!pip install -e .
!pip install ninja
!pip install flash-attn --no-build-isolation

### Clone Vicuna
!git clone https://huggingface.co/lmsys/vicuna-7b-v1.3

### Download Data
!gdown https://drive.google.com/u/0/uc?id=1a9XB3r83ZCFWLOHBp8ooz3zQFl9rEIei&export=download

### Preprocessing
%cd /content
!git clone https://github.com/pimang62/dacon-multimodal-vqa.git

%cd /content/dacon-multimodal-vqa
!python preprocessing.py

## 4. Run (put your API)
%cd /content/LLaVA
!pip install wandb
!wandb login

### Train
!python /content/LLaVA/llava/train/trian_mem.py \
    --model_name_or_path /content/LLaVA/vicuna-7b-v1.3 \
    --version v1 \
    --data_path /content/output.json \
    --image_folder /content/data/image/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir /content/drive/MyDrive/LLaVA checkpoint \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 128 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb

## 5. Re-training
'''
- You should put 'vicuna' to your model-name
- output_dir folder should be contained 'checkpoint-*'
- num_train_epochs must have started from 2 or more
'''
!python /content/LLaVA/llava/train/train_mem.py\
    --model_name_or_path /content/LLaVA/vicuna-7b-v1.3 \
    --version v1 \
    --data_path /content/output.json \
    --image_folder /content/data/image/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir /content/driver/MyDrive/LLaVA/checkpoint-2400 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.00 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 128 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb

## 6. Inference
'''
You should change output_dir name 'hecckpoint'-to 'llava-'
    May be you might get a difference whether the name contains 'llava' or not
'''
%cd /content/LLaVA
!python /content/dacon-multimodal-vqa/eval/modeel_vqa.py \
    --model-path /content/drive/MyDrive/LLaVA checkpoint/LLaVA-7B-v1.3 \
    --model_base lmsys/vicuna-7b-v1.3 \
    --question_file \
    /content/dacon-multimodal-vqa/test.jsonl \
    --image-folder \
    /content/image/test \
    --answers-file \
    /content/result.jsonl \
    
## 7. Submission
%cd /content/dacon-multimodal-vqa
!python submission.py


## preprocessing.py
## https://github.com/geuk-hub/-Dacon-Multimodal-vqa/blob/main/preprocessing.py

import zipfile
import os
import csv
import json

### unzip
%cd /content

!unzip --qq "/content/drive/MyDrive/data.zip"

### --------------------------------------------------------------
### make 'ouptut.json'
with open('/content/data/train.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    data = list(reader)

json_data = []
for row in data:
    id, image_id, question, answer = row
    json_data.append({
        "id": id,
        "image": "/content/data/image/train/" + image_id + ".jpg",
        "conversations" : [
            {
                "from": "human",
                "value": "<image>\n" + question
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    })

with open('output.json', 'w') as f:
    json.dum(json_data, f, indent=4)

### --------------------------------------------------------------
### make 'test.json'
with open('content/data/test.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    data = list(reader)

json_data = []
for row in data:
    id, image_id, qeustion = row
    json_data.append({
        "id": id,
        "image": "/content/data/image/test" + image_id + ".jpg",
        "text": question
    })

### jsonl file path
jsonl_output_file = "/content/test.jsonl"

### JSON to JSONL
with open(jsonl_output_file, 'w') as file:
    for obj in json_data:
        # write file (JSON + (\n)).
        json.dump(obj, file)
        file.write("\n")


## builder.py
## https://github.com/geuk-hub/-Dacon-Multimodal-vqa/blob/main/model/builder.py
'''
Copyright 2023 Haotian Liu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig    
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
            bnt_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype']j = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVa model
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model =LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'nono_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id = repo_id,
                        filename = filename,
                        subfolder=subfolder)
                    return torch.laod(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.starswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k) : v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is laded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                model.resize_token_embeddings(32003)    # edit

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    
    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device='cuda', dtype=torch.float16)
        image_processor = vision_tower.image_processor
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


## model_vqa.py
## https://github.com/geuk-hub/-Dacon-Multimodal-vqa/blob/main/eval/model_vqa.py
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model     # edit
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)    # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)] 

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.questionJ_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line['id']    # edit
        image_file = line['image']
        qs = line['text']
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor =image_processor.preprocessor(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images = image_tensor.unsqueeze(0).half().cuda(),
                do_sample = True,
                temperature = args.temperature,
                top_p = args.top_p,
                num_beams = args.num_beams,
                # no_repeat_ngram_size = 3, 
                max_new_tokens = 1024,
                use_cache = True)
        
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "text": outputs}) + "\n")    # edit
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="" )
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2 )
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1 )
    args = parser.parse_args()

    eval_model(args)


## submission.py
## https://github.com/geuk-hub/-Dacon-Multimodal-vqa/blob/main/submission.py
import pandas as pd
submission = pd.read_csv('/content/data/sample_submission.csv')

import json
with open('/content/result.jsonl', 'r') as file:
    answer = []
    for line in file:
        info = json.loads(line)
        answer.append(info['text'].strip())

submission['answer'] = pd.DataFrame(answer)
submission.to_csv('LLaVA_submission.csv', index=False)