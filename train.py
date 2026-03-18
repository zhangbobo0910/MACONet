#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py  （已集成“自动选择显存最大的 GPU”功能）

执行示例：
    python train.py -d ace2004 -b 4 --seed 13
"""

import os
import warnings
import argparse
import json
import shutil
import random
import subprocess
from pathlib import Path

# ============ 0. 自动检测显存最大的 GPU ============ #
def pick_best_gpu() -> str | None:
    """
    返回“空闲显存最多”的 GPU 的编号（字符串），
    若检测失败或无 GPU，则返回 None。
    优先使用 pynvml；失败时 fallback 到 nvidia-smi。
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        gpus = []
        for idx in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = mem.free // 1024**2
            gpus.append((free_mb, idx))
        pynvml.nvmlShutdown()
        if gpus:
            gpus.sort(reverse=True)          # free_mb 从大到小
            return str(gpus[0][1])
    except Exception:
        # fall-back 到 nvidia-smi
        try:
            cmd = r"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
            out = subprocess.check_output(cmd.split()).decode().strip().splitlines()
            if out:
                free_list = [(int(mb), idx) for idx, mb in enumerate(out)]
                free_list.sort(reverse=True)
                return str(free_list[0][1])
        except Exception:
            pass
    return None


BEST_GPU = pick_best_gpu()
if BEST_GPU is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = BEST_GPU
    print(f"[Info] Picked GPU {BEST_GPU} (most free memory).")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("[Info] No GPU detected or query failed, fallback to CPU.")

# ============ 1. 常规环境变量与缓存目录 ============ #
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MKL_THREADING_LAYER"]    = "GNU"

# huggingface 缓存
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
os.environ["HF_HOME"]            = "/tmp/huggingface_cache"
Path("/tmp/huggingface_cache").mkdir(parents=True, exist_ok=True)
try:
    import getpass, pwd, grp
    current_user = getpass.getuser()
    shutil.chown("/tmp/huggingface_cache", user=current_user, group=current_user)
except Exception:
    # 改权限失败就给 777，保证可写
    os.chmod("/tmp/huggingface_cache", 0o777)

warnings.filterwarnings("ignore")

# ============ 2. 之后的包 import 与业务逻辑保持不变 ============ #
import numpy as np
import torch

from fastNLP.core.callbacks.topk_saver import TopkSaver
from fastNLP import (
    cache_results, prepare_torch_dataloader, print, Trainer,
    TorchGradClipCallback, FitlogCallback, CheckpointCallback,
    SortedSampler, BucketedBatchSampler, TorchWarmupCallback
)
import fitlog

from model.model    import CNNNer
from model.metrics  import NERMetric
from data.ner_pipe  import SpanNerPipe
from data.padder    import Torch3DMatrixPadder

# ---------------- 参数 ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--encoder_lr", default=2e-5, type=float)
parser.add_argument("-b", "--batch_size", default=2, type=int)
parser.add_argument("-n", "--n_epochs", default=50, type=int)
parser.add_argument("--warmup", default=0.1, type=float)
parser.add_argument("-d", "--dataset_name", default="ace2004", type=str)  # ace2005 genia ace2004
parser.add_argument("--model_name", default=None, type=str)
parser.add_argument("--cnn_depth", default=1, type=int)
parser.add_argument("--cnn_dim", default=120, type=int)
parser.add_argument("--num", default=1, type=int)
parser.add_argument("--logit_drop", default=0.0, type=float)
parser.add_argument("--biaffine_size", default=200, type=int)
parser.add_argument("--n_head", default=5, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--n_layer", default=1, type=int)
parser.add_argument("--accumulation_steps", default=1, type=int)
parser.add_argument("--separateness_rate", default=5, type=int)
parser.add_argument("--theta", default=1.0, type=float)
parser.add_argument("--loss_theta", default=1.0, type=float)
args = parser.parse_args()
dataset_name = args.dataset_name

# 自动映射模型
if args.model_name is None:
    args.model_name = {
        "genia"   : "dmis-lab/biobert-v1.1",
        "conll03" : "bert-large-cased",
        "ace2004" : "roberta-base",
        "ace2005" : "roberta-base"
    }.get(dataset_name, "roberta-base")
print(f"Using model `{args.model_name}` for dataset `{dataset_name}`")

# ---------------- 复现性 ---------------- #
def seed_torch(seed=43):
    import os, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

fitlog.set_log_dir("logs/")
seed = fitlog.set_rng_seed(args.seed)
seed_torch(seed)
os.environ["FASTNLP_GLOBAL_SEED"] = str(seed)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)

# ============ 3. 数据准备 ============ #
@cache_results("caches/ner_caches.pkl", _refresh=False)
def get_data(dataset_name, model_name):
    paths_map = {
        "ace2004": "preprocess/outputs/ace2004",
        "ace2005": "preprocess/outputs/ace2005",
        "genia"  : "preprocess/outputs/genia",
        "conll03": "preprocess/outputs/conll03",
    }
    if dataset_name not in paths_map:
        raise RuntimeError("Unsupported dataset.")
    pipe = SpanNerPipe(model_name=model_name)
    return pipe.process_from_file(paths_map[dataset_name]), pipe.matrix_segs

dl, matrix_segs = get_data(dataset_name, args.model_name)

# 稠密化 matrix
def densify(x):
    return x.todense().astype(np.float32)

dl.apply_field(densify, field_name="matrix", new_field_name="matrix", progress_bar="Densify")
print(dl)
label2idx = getattr(dl, "ner_vocab", getattr(dl, "label2idx"))
print(f"{len(label2idx)} labels: {label2idx}, matrix_segs:{matrix_segs}")

dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad(
        "matrix",
        pad_fn=Torch3DMatrixPadder(
            pad_val=ds.collator.input_fields["matrix"]["pad_val"],
            num_class=matrix_segs["ent"],
            batch_size=args.batch_size,
        ),
    )
    if name == "train":
        _dl = prepare_torch_dataloader(
            ds,
            batch_size=args.batch_size,
            batch_sampler=BucketedBatchSampler(
                ds, "input_ids", batch_size=args.batch_size, num_batch_per_bucket=30
            ),
            pin_memory=True,
            shuffle=True,
            num_workers=0,
            prefetch_factor=None,
        )
    else:
        _dl = prepare_torch_dataloader(
            ds,
            batch_size=args.batch_size,
            sampler=SortedSampler(ds, "input_ids"),
            pin_memory=True,
            shuffle=False,
            num_workers=0,
            prefetch_factor=None,
        )
    dls[name] = _dl

# ============ 4. 构建模型 ============ #
non_ptm_lr_ratio = 100
weight_decay     = 1e-4
size_embed_dim   = 25
ent_thres        = 0.5
kernel_size      = 3
n_head           = args.n_head

model = CNNNer(
    args.model_name,
    num_ner_tag=matrix_segs["ent"],
    cnn_dim=args.cnn_dim,
    biaffine_size=args.biaffine_size,
    size_embed_dim=size_embed_dim,
    logit_drop=args.logit_drop,
    n_layer=args.n_layer,
    kernel_size=kernel_size,
    n_head=n_head,
    cnn_depth=args.cnn_depth,
    separateness_rate=args.separateness_rate / 100.0,
    theta=args.theta,
)

# 统计参数
import collections
counter = collections.Counter()
for name, param in model.named_parameters():
    counter[name.split(".")[0]] += param.numel()
print(counter)
print("Total param", sum(counter.values()))
fitlog.add_to_line(json.dumps(counter, indent=2))
fitlog.add_other(value=sum(counter.values()), name="total_param")

# 参数分组
ln_params, non_ln_params = [], []
non_pretrain_ln_params, non_pretrain_params = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    low_name = name.lower()
    if "pretrain_model" in low_name:
        (ln_params if ("norm" in low_name or "bias" in low_name) else non_ln_params).append(param)
    else:
        (non_pretrain_ln_params if ("norm" in low_name or "bias" in low_name) else non_pretrain_params).append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": non_ln_params,          "lr": args.lr,                        "weight_decay": weight_decay},
        {"params": ln_params,              "lr": args.lr,                        "weight_decay": 0},
        {"params": non_pretrain_ln_params, "lr": args.lr * non_ptm_lr_ratio,     "weight_decay": 0},
        {"params": non_pretrain_params,    "lr": args.lr * non_ptm_lr_ratio,     "weight_decay": weight_decay},
    ]
)

# ============ 5. callbacks ============ #
callbacks = [
    FitlogCallback(log_loss_every=20),
    CheckpointCallback(monitor="f#f#test", save_evaluate_results=True, folder="_saved_models", topk=3),
    TorchGradClipCallback(clip_value=5),
    TorchWarmupCallback(warmup=args.warmup, schedule="linear"),
]

evaluate_dls = {k: v for k, v in dls.items() if k in {"dev", "test"}}
metrics = {"f": NERMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, allow_nested=True)}

# device index 在 CUDA_VISIBLE_DEVICES 内部重新编号，因此这里传 0 即使用所选 GPU
device_id = 0 if torch.cuda.is_available() else "cpu"

trainer = Trainer(
    model=model,
    driver="torch",
    train_dataloader=dls.get("train"),
    evaluate_dataloaders=evaluate_dls,
    optimizers=optimizer,
    callbacks=callbacks,
    overfit_batches=0,
    device=device_id,
    n_epochs=args.n_epochs,
    metrics=metrics,
    monitor="f#f#dev",
    evaluate_every=-1,
    evaluate_use_dist_sampler=True,
    accumulation_steps=args.accumulation_steps,
    fp16=False,
    progress_bar="rich",
)

trainer.run(num_train_batch_per_epoch=-1, num_eval_batch_per_dl=-1, num_eval_sanity_batch=1)
fitlog.finish()
