# built-in library
import os
import os.path as osp
import json

# torch
import torch
from torch.utils.data import Dataset

# external library
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import GroupKFold


class XRayDataset(Dataset):
    def __init__(self, pngs, jsons, settings, is_train=True, transforms=None):
        """학습에 사용할 XRay 데이터셋을 정의한 클래스입니다.

        Args:
            is_train (bool, optional): train인지 아닌지 판단하는 변수입니다. Defaults to True.
            transforms (_type_, optional): 이미지에 적용할 transformations입니다. Defaults to None.
        """
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)

        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [osp.dirname(fname) for fname in _filenames]

        # dummy label
        ys = [0 for _ in _filenames]

        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue

                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])

                # skip i > 0
                break
        
        self.settings = settings
        self.class2ind = {v: i for i, v in enumerate(self.settings['classes'])}
        self.ind2class = {v: k for k, v in self.class2ind.items()}

        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = osp.join(self.settings['image_root'], image_name)

        image = cv2.imread(image_path)
        image = np.array(image, dtype=np.float32) / 255.

        label_name = self.labelnames[item]
        label_path = osp.join(self.settings['label_root'], label_name)

        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(self.settings['classes']), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = self.class2ind[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class XRayInferenceDataset(Dataset):
    def __init__(self, pngs, settings, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.settings = settings
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = osp.join(self.settings['tt_image_root'], image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tensor will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()
            
        return image, image_name