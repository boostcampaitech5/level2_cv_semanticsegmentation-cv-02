# built-in library
from tqdm import tqdm

# external library
import wandb

# torch
import torch
import torch.nn.functional as F

# custom modules
from metrics import dice_coef


def validation(settings, device, epoch, model,
               data_loader, criterion, num_val_batches, thr=0.5):
    """학습한 모델을 평가할 때 사용하는 함수입니다.

    Args:
        settings (_type_): evaluation에 필요한 기본 설정값들이 담겨있는 dictionary 입니다.
        device (_type_): 연산에 사용할 장치입니다. (gpu or cpu)
        epoch (_type_): evaluation이 진행되고 있는 현재 epoch을 출력하기 위해 사용합니다.
        model (_type_): evaluation 할 모델입니다.
        data_loader (_type_): evaluation에 사용할 dataloader 입니다.
        criterion (_type_): validation loss 계산에 사용할 loss 함수입니다.
        num_val_batches (_type_): evaluation 중 mean epoch loss를 연산하기 위해 필요한 총 batch의 개수입니다.
        thr (float, optional): threshold를 기준으로 이진화를 수행합니다. Defaults to 0.5.

    Returns:
        _type_: 모든 클래스에 대해 연산 후 평균을 취한 average dice coefficients를 반환합니다.
    """
    
    print(f"Start validation #{epoch:2d}")
    model.eval()

    dices = []
    # num_class = len(settings['classes'])
    total_loss = 0
    cnt = 0

    with torch.no_grad():
        for images, masks in tqdm(data_loader, total=len(data_loader)):
            images, masks = images.to(device), masks.to(device)

            if settings['lib'] == 'smp':
                outputs = model(images)
            else:
                outputs = model(images)['out']

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(settings['classes'], dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    # wandb 기록 - validation
    wandb.log({
        "Epoch": epoch,
        "Valid/Mean_Epoch_Loss": round(total_loss / num_val_batches, 4),
        "Valid/Average_Dice_Coef": round(avg_dice, 4),
        "Valid/finger-16": round(dices_per_class[15].item(), 4),
        "Valid/Trapezoid": round(dices_per_class[20].item(), 4),
        "Valid/Pisiform": round(dices_per_class[26].item(), 4)
    })

    return avg_dice
