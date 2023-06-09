# python native
import os
import os.path as osp
import math
from pprint import pprint

# external library
import albumentations as A
from argparse import ArgumentParser
import wandb

# torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# custom modules
from utils import load_config, get_exp_name, set_seed, sep_cfgs
from my_trainer import fcn_trainer
from my_dataset import XRayDataset
from my_models import MyModels

# visualization
# import matplotlib.pyplot as plt


def parse_args():
    """training, evaluation에 필요한 yaml 파일의 경로를 가져오기 위해 사용합니다.

    Returns:
        _type_: 사용자가 입력한 arguments를 반환합니다.
    """
    parser = ArgumentParser()

    parser.add_argument('--config_path', type=str, default='../configs/baseline.yaml', help='yaml files to train segmentation models (default: ../configs/baseline.yaml)')

    args = parser.parse_args()

    return args


def main(args):
    """모델 학습을 위해 필요한 값들을 정의하고, training 함수를 호출하기 위해 사용합니다.

    Args:
        args (_type_): 사용자가 직접 입력한 arguments입니다.
    """
    # yaml 파일 불러오기
    configs = load_config(args.config_path)
    pprint(configs)

    # wandb에 config 업로드하기
    wandb.config.update(configs)

    # wandb 실험 이름 설정
    run_name = get_exp_name(args.config_path)
    wandb.run.name = run_name

    # sep_cfgs 함수를 이용하여, 사용하기 쉽게 분리
    settings, train_cfg, val_cfg, _ = sep_cfgs(configs)

    # seed 세팅
    set_seed(settings['seed'])

    # dataset 생성에 필요한 pngs, jsons 만들기
    pngs = {
        osp.relpath(osp.join(root, fname), start=settings['image_root'])
        for root, _dirs, files in os.walk(settings['image_root'])
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=settings['label_root'])
        for root, _dirs, files in os.walk(settings['label_root'])
        for fname in files
        if osp.splitext(fname)[1].lower() == ".json"
    }
    jsons_fn_prefix = {osp.splitext(fname)[0] for fname in jsons}
    pngs_fn_prefix = {osp.splitext(fname)[0] for fname in pngs}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    pngs = sorted(pngs)
    jsons = sorted(jsons)

    # 데이터셋 정의
    tf = A.Resize(512, 512)

    train_dataset = XRayDataset(pngs, jsons, settings, is_train=True, transforms=tf)
    valid_dataset = XRayDataset(pngs, jsons, settings, is_train=False, transforms=tf)

    num_train_batches = math.ceil(len(train_dataset) / train_cfg['batch_size'])
    num_val_batches = math.ceil(len(valid_dataset) / val_cfg['batch_size'])

    # define dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=train_cfg['shuffle'],
        num_workers=train_cfg['num_workers'],
        drop_last=train_cfg['drop_last']
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=val_cfg['batch_size'],
        shuffle=val_cfg['shuffle'],
        num_workers=val_cfg['num_workers'],
        drop_last=val_cfg['drop_last']
    )

    # set model, optimizer, loss function
    my_model = MyModels(settings)
    model = getattr(my_model, train_cfg['models'])() # getattr 함수를 사용하여 cfg로만 모델을 불러올 수 있도록 함

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])

    # 실험 시작
    fcn_trainer(run_name, settings, train_cfg, val_cfg,
                model, train_loader, valid_loader, criterion, optimizer,
                num_train_batches, num_val_batches)


if __name__ == '__main__':
    # wandb 세팅
    wandb.init(project="Hand Bone Segmentation", reinit=True)

    # config_path 불러오기
    args = parse_args()
    print(f"config_path : {args.config_path}")

    # 실험 시작
    main(args)
    