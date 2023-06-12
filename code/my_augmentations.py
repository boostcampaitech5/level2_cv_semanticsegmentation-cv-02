# external-library
import albumentations as A


class MyAugs():
    def resize_512(self):
        tf = A.Resize(512, 512)

        return tf
    
    def resize_1024(self):
        tf = A.Resize(1024, 1024)

        return tf
    
    def resize_512_imagenet_normalize(self):
        tf = A.Compose([
            A.Resize(512, 512),
            A.Normalize()
        ])

        return tf
    
    def dh_best_augs(self):
        """fcn_resnet50 모델에서 lb 0.9679를 기록한 augmentations
        batch: 4, lr: 0.003, using lr_scheduler        
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20, 40, 60, 80], gamma=0.7)

        Returns:
            _type_: _description_
        """
        tf = A.Compose([
            A.Resize(1024, 1024),
            A.Rotate(limit=50, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GridDropout(ratio=0.2, random_offset=True,
                          holes_number_x=4, holes_number_y=4, p=0.5)
        ])

        return tf