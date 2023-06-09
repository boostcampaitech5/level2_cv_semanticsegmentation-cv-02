# torch
from torchvision import models
import torch.nn as nn


class MyModels():
    """
    getattr() 함수와 같이 사용해서, yaml 파일의 model value만 변경하면
    training code 수정 없이 실험을 진행할 수 있도록 만들어주는 클래스입니다.

    새로운 모델을 추가하고 싶다면 아래의 클래스 함수들처럼 만들어주시면 됩니다.

    클래스 함수의 이름은 yaml 파일의 model value로 사용되고,
    getattr() 함수를 통해 model을 불러오게 됩니다.

    **주의사항**
    -> 모든 클래스 함수들은 모델을 반환해야만 하며, 모델의 출력이 fcn과
    다른 경우 my_trainer.py에서 새로운 trainer 함수를 만들어주어야 합니다.
    """
    def __init__(self, settings):
        self.num_class = len(settings['classes'])

    def fcn_resnet50(self):
        model = models.segmentation.fcn_resnet50(pretrained=True)

        model.classifier[4] = nn.Conv2d(512, self.num_class, kernel_size=1)

        return model
    
    def fcn_resnet101(self):
        model = models.segmentation.fcn_resnet101(pretrained=True)

        model.classifier[4] = nn.Conv2d(512, self.num_class, kernel_size=1)

        return model
    
    def deeplabv3_resnet50(self):
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)

        model.classifier[4] = nn.Conv2d(256, self.num_class, kernel_size=1)

        return model
    
    def deeplabv3_resnet101(self):
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)

        model.classifier[4] = nn.Conv2d(256, self.num_class, kernel_size=1)

        return model
    