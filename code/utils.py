# torch
import torch

# external library
import numpy as np

# built-in library
import random
import os
import os.path as osp
import yaml

# define colors
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]


def set_seed(seed: int=21):
    """실험의 재현 가능성을 위해 시드를 설정할 때 사용하는 함수입니다.

    Args:
        seed (int, optional): 시드로 사용할 값. Defaults to 21.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, save_dir, file_name='fcn_resnet50_best_model.pt'):
    """학습한 모델을 저장할 때 사용하는 함수입니다.

    Args:
        model (_type_): 학습이 완료된 모델
        save_dir (_type_): 모델을 저장할 디렉토리 경로
        file_name (str, optional): 저장할 모델의 이름. Defaults to 'fcn_resnet50_best_model.pt'.
    """
    if not osp.exists(save_dir):
        os.mkdir(save_dir)

    output_path = osp.join(save_dir, file_name)
    
    torch.save(model, output_path)


def load_config(config_file):
    """정의한 YAML 파일을 불러오는 함수입니다.

    Args:
        config_file : 실험에 필요한 설정들을 저장해둔 yaml 파일
    """
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config


def get_exp_name(config_path: str):
    """현재 진행 중인 실험의 이름을 설정하는 함수입니다.
    yaml 파일의 이름을 바탕으로 설정됩니다.
    ex) config_path: ../configs/sy/01_baseline.yaml -> 01_baseline

    Args:
        config_path (str): yaml config 파일의 경로입니다.
    """

    exp_name = config_path.split('/')[-1].split('.')[0]

    return exp_name


def sep_cfgs(configs):
    """가져다 쓰기 쉽게 configs 파일을 분리하는 함수입니다.

    Args:
        configs (_type_): 불러온 yaml 파일의 정보가 담겨있는 dictionary입니다.

    Returns:
        _type_: 분리한 configs 파일을 반환합니다.
    """
    return configs['settings'], configs['train'], configs['val'], configs['test']

  
def custom_collate_fn(sample):
    img, label = list(zip(*sample))
    
    img = np.array(img, dtype=np.float32)
    label = np.array(label, dtype=np.float32)

    return img, label

  
# utility function
# this does not care overlap
def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image
