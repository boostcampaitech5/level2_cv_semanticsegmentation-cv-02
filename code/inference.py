# built-in library
import os
import os.path as osp
from argparse import ArgumentParser
from pprint import pprint
from tqdm import tqdm

# external library
import numpy as np
import pandas as pd
import albumentations as A

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# custom library
from my_dataset import XRayInferenceDataset
from my_models import MyModels
from utils import load_config, sep_cfgs


def encode_mask_to_rle(mask):
    """mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.

    Args:
        mask (_type_): numpy array binary mask (1: mask, 0: background)

    Returns:
        _type_: encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    """RLE로 인코딩된 결과를 mask map으로 복원합니다.

    Args:
        rle (_type_): _description_
        height (_type_): _description_
        width (_type_): _description_

    Returns:
        _type_: _description_
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


def parse_args():
    """inference에 필요한 yaml 파일의 경로 및 model을 가져오기 위해 사용합니다.

    Returns:
        _type_: 사용자가 입력한 arguments를 반환합니다.
    """
    parser = ArgumentParser()

    parser.add_argument('--config_path', type=str, default='../configs/baseline.yaml', help='yaml files to test a segmentation model (default: ../configs/baseline.yaml)')
    parser.add_argument('--model_path', type=str, default='../trained_models/fcn_resnet50_best.pth', help='model weight path (default: ../trained_models/fcn_resnet50_best.pth)')

    args = parser.parse_args()

    return args


def test(settings, model, data_loader, thr=0.5):
    CLASS2IND = {v: i for i, v in enumerate(settings['classes'])}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(settings['classes'])

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)['out']
            
            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class


def main(configs):
    settings, train_cfg, _, test_cfg = sep_cfgs(configs)

    pngs = {
        osp.relpath(osp.join(root, fname), start=settings['tt_image_root'])
        for root, _dirs, files in os.walk(settings['tt_iamge_root'])
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }

    tf = A.Resize(512, 512)

    test_dataset = XRayInferenceDataset(pngs, settings, transforms=tf)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=test_cfg['batch_size'],
                             shuffle=test_cfg['shuffle'],
                             num_workers=test_cfg['num_workers'],
                             drop_last=test_cfg['drop_last'])

    my_model = MyModels(settings)    
    model = getattr(my_model, train_cfg['models'])()

    rles, filename_and_class = test(model, test_loader)

    # To CSV
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    if not osp.exists(settings['submission_dir']):
        os.mkdir(settings['submission_dir'])
    
    submission_filename = osp.join(settings['submission_dir'], train_cfg['models'] + '.csv')
    df.to_csv(submission_filename, index=False)


if __name__ == "__main__":
    args = parse_args()
    print(f"config_path : {args.config_path}")

    # yaml 파일 불러오기
    cfgs = load_config(args.config_path)
    pprint(cfgs)

    # inference
    main(cfgs)
