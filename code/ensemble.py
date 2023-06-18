# built-in library
import os
import os.path as osp
from argparse import ArgumentParser
from pprint import pprint
from tqdm import tqdm
import random
import cv2
import matplotlib.pyplot as plt

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
from my_dataset import ensemble_XRayInferenceDataset
from my_models import MyModels
from my_augmentations import MyAugs
from utils import load_config, sep_cfgs, get_exp_name, label2rgb, set_seed


def parse_args():
    """inference에 필요한 yaml 파일의 경로 및 model을 가져오기 위해 사용합니다.

    Returns:
        _type_: 사용자가 입력한 arguments를 반환합니다.
    """
    parser = ArgumentParser()
    parser.add_argument('--models_txt_path', type=str, help="ensemble models.txt path.  if you choose option 1, models_txt_path is required")
    parser.add_argument('--save_dir', type=str, default="/opt/ml", help="save dir")
    parser.add_argument('--file_name', type=str, default="ensemble_ex1", help="save csv file name")
    parser.add_argument('--img_size', type=int, default=1024, help="transform image size")
    parser.add_argument('--thr', type=float, default=0.5, help="threshold")
    args = parser.parse_args()

    return args


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


def ensemble_test(models_txt_path, data_loader, thr=0.5):
    CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]

    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rles = []
    filename_and_class = []
    models = []
    with open(models_txt_path, 'r') as f:
        for line in f:
            library , model_path = line.strip().split(' ')
            models.append([library, model_path])
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            
            outputs_list = []
            for library, model_path in models:
                model = torch.load(model_path).cuda()
                model.eval()
                if library == "smp":  #smp라면
                    outputs = model(images)
                else: #그 외
                    outputs = model(images)
                outputs_list.append(outputs)
            
            # 앙상블 방식을 적용하여 outputs 결합
            ensemble_outputs = torch.mean(torch.stack(outputs_list), dim=0)
            
            ensemble_outputs = F.interpolate(ensemble_outputs, size=(2048, 2048), mode="bilinear")
            ensemble_outputs = torch.sigmoid(ensemble_outputs)
            ensemble_outputs = (ensemble_outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(ensemble_outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class


def visualization(rles,  filename_and_class, args, IMAGE_ROOT):

    #5개 이미지 랜덤 샘플링
    image_len = [i for i in range(len(filename_and_class)//29)]
    image_num = random.sample(image_len, 5)

    SAVE_PATH = f'../example_image/{args.file_name}'

    if not osp.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for idx in image_num:
        image = cv2.imread(osp.join(IMAGE_ROOT, filename_and_class[idx*29].split("_")[1]))
        preds = []
        
        for rle in rles[idx*29: idx*29+29]:
            pred = decode_rle_to_mask(rle, height=2048, width=2048)
            preds.append(pred)

        preds = np.stack(preds, 0)

        fig, ax = plt.subplots(1, 2, figsize=(24, 12))
        ax[0].imshow(image)    # remove channel dimension
        ax[1].imshow(label2rgb(preds))

        plt.savefig(osp.join(SAVE_PATH, f'{idx}.png'))



def main(args):
    IMAGE_ROOT = "/opt/ml/input/data/test/DCM"
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    tf = A.Resize(args.img_size, args.img_size) 
    test_dataset = ensemble_XRayInferenceDataset(pngs, IMAGE_ROOT, transforms=tf)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    rles, filename_and_class = ensemble_test(args.models_txt_path, test_loader, args.thr)
        
    # 시각화
    visualization(rles, filename_and_class, args, IMAGE_ROOT)

    # To CSV   
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [osp.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    submission_dir, file_name = args.save_dir, args.file_name

    if not osp.exists(submission_dir):
        os.mkdir(submission_dir)

    submission_filename = osp.join(submission_dir,file_name+'.csv')
    
    if not osp.exists(args.save_dir):
        os.mkdir(args.save_dir)
    

    df.to_csv(submission_filename, index=False)
    

if __name__ == "__main__":
    args = parse_args()

    # inference
    main(args)
