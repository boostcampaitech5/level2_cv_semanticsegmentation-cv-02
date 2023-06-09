# torch
import torch

# external library
import wandb

# custom library
from evaluation import validation
from utils import save_model


# TODO: ETA 추가
def fcn_trainer(save_file_name, settings, train_cfg, val_cfg,
                model, train_loader, val_loader, criterion, optimizer,
                num_train_batches, num_val_batches):
    """baseline으로 제공된 fcn 모델 학습을 위한 trainer 함수입니다.

    Args:
        save_file_name (_type_): 학습한 모델을 저장할 때 사용할 이름입니다.
        settings (_type_): 학습 시 설정할 기본 값들이 담겨있는 dictionary 입니다.
        train_cfg (_type_): 모델 학습에 사용할 설정 값들이 담겨있는 dictionary 입니다.
        val_cfg (_type_): evaluation에 사용할 설정 값들이 담겨있는 dictionary 입니다.
        model (_type_): 학습할 모델입니다.
        train_loader (_type_): train dataloader입니다.
        val_loader (_type_): valid dataloader입니다.
        criterion (_type_): loss 함수입니다.
        optimizer (_type_): 최적화 시 사용할 함수입니다.
        num_train_batches (_type_): training 중 mean epoch loss를 연산하기 위해 필요한 총 batch의 개수입니다.
        num_val_batches (_type_): evaluation 중 mean epoch loss를 연산하기 위해 필요한 총 batch의 개수입니다.
    """
    # device setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Start training..")
    model = model.to(device)

    # num_class = len(settings['classes'])
    best_dice = 0.

    # training loop
    for epoch in range(train_cfg['num_epochs']):
        model.train()

        total_loss = 0.
        for step, (images, masks) in enumerate(train_loader):
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            # forward
            outputs = model(images)['out']

            # loss 계산 & update
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # log_interval에 따라 training 과정 중 연산된 값들을 출력
            if (step + 1) % train_cfg['log_interval'] == 0:
                print(
                    f'Epoch [{epoch + 1}/{train_cfg["num_epochs"]}], '
                    f'Step [{step + 1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(), 4)}'
                )

        # wandb 기록 - training
        wandb.log({
            "Epoch": epoch + 1,
            "Train/Mean_Epoch_Loss": round(total_loss / num_train_batches, 4),
        })
        
        # evaluation
        if (epoch + 1) % val_cfg['val_every'] == 0: # val_every에 따라 evaluation을 수행
            dice = validation(settings, device, epoch + 1,
                              model, val_loader, criterion, num_val_batches) # wandb 기록은 validation 함수 내부에서 진행됨

            # 이번 에폭 dice가 기존 best dice보다 높으면 모델을 저장
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {settings['saved_dir']}")                
                best_dice = dice

                filename = save_file_name + "_best.pth" # 학습한 모델의 이름으로 best.pth를 저장함
                save_model(model, settings['saved_dir'], filename)
