# torch
from torchvision import models
import torch.nn as nn

# external-library
import segmentation_models_pytorch as smp


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

    # torchvision
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
    
    # smp (https://github.com/qubvel/segmentation_models.pytorch)
    def pspnet_resnet50(self):
        model = smp.PSPNet(
            encoder_name="resnet50",
            encoder_depth=5,
            encoder_weights="imagenet",
            in_channels=3,
            classes=29,
        )
    
        model.segmentation_head = nn.Sequential(
            nn.Conv2d(512, self.num_class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.UpsamplingBilinear2d(scale_factor=32.0),
            nn.Identity()
        )

        return model
   
    def deeplabv3plus_xception41(self):
        model = smp.DeepLabV3Plus(
            encoder_name='tu-xception41',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=self.num_class)
        
        return model

     def deeplabv3plus_xception41p(self):
        model = smp.DeepLabV3Plus(
            encoder_name='tu-xception41p',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=29
        )

        return model

    def deeplabv3plus_xception65(self):
        model = smp.DeepLabV3Plus(
            encoder_name='tu-xception65',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=self.num_class)
          
        return model

    def unet_resnet50(self):
        model = smp.Unet(
            encoder_name="resnet50",        
            encoder_weights="imagenet",    
            in_channels=3,                  
            classes=self.num_class,     
        )

        return model
      
    def unet_resnet101(self):
        model = smp.Unet(
            encoder_name="resnet101",        
            encoder_weights="imagenet",    
            in_channels=3,                  
            classes=self.num_class,     
        )

        return model
        
    def unet_efficientnetb4(self):
        model = smp.Unet(
            encoder_name="efficientnet-b4",        
            encoder_weights="imagenet",    
            in_channels=3,                  
            classes=self.num_class,     
        )

        return model

    def unet_mitb4(self):
        model = smp.Unet(
            encoder_name="mit_b4",        
            encoder_weights="imagenet",    
            in_channels=3,                  
            classes=self.num_class,     
        )

        return model

    def unetplusplus_resnet50(self):
        model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",    
            in_channels=3,                  
            classes=self.num_class,   
        )
        return model

    def manet_resnet50(self):
        model = smp.MAnet(
            encoder_name="resnet50",
            encoder_weights="imagenet",    
            in_channels=3,                  
            classes=self.num_class,   
        )

        return model