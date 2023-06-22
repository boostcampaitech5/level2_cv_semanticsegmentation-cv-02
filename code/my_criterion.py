# torch
import torch
import torch.nn.functional as F
import torch.nn as nn


class MyCriterion():
    def __init__(self):
        pass

    def bce_with_logit_loss(self, inputs, targets):
        criterion = nn.BCEWithLogitsLoss()

        return criterion(inputs, targets)

    def focal_loss(self, inputs, targets, alpha=.25, gamma=2) : 
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        loss = alpha * (1-BCE_EXP)**gamma * BCE

        return loss 

    def IOU_loss(self, inputs, targets, smooth=1) : 
        inputs = F.sigmoid(inputs)      
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        IoU = (intersection + smooth)/(union + smooth)
        
        return 1 - IoU

    def dice_loss(self, pred, target, smooth=1.) :
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))

        return loss.mean()

    def combined_loss(self, pred, target, weight=0.5) :
        focal = self.focal_loss(pred, target)
        pred = F.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        loss = focal * weight + dice*(1-weight)

        return loss
    
    def bce_and_dice_loss(self, pred, target, weight=0.3):
        bce = self.bce_with_logit_loss(pred, target)
        
        pred = F.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        
        loss = bce * weight + dice * (1 - weight)

        return loss