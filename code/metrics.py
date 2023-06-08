# torch
import torch


def dice_coef(y_true, y_pred):
    """dice coefficient를 계산하는 함수입니다.

    Args:
        y_true (_type_): ground truth
        y_pred (_type_): model prediction

    Returns:
        _type_: 계산된 dice score를 반환합니다.
    """
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)