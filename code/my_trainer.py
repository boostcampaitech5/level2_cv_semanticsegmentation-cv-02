# torch
import torch

# external library
import wandb

# built-in library
import os
from collections import deque

# custom library
from evaluation import validation
from utils import save_model


class MyTrainer():
    def __init__(self, save_dir_name, settings, train_cfg, val_cfg,
                 model, train_loader, val_loader, criterion, optimizer, scheduler,
                 num_train_batches, num_val_batches):
        """모델 학습에 필요한 arguments를 입력받습니다.

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
            scheduler (_type_): 학습 시 사용할 learning rate scheduler입니다.
            num_train_batches (_type_): training 중 mean epoch loss를 연산하기 위해 필요한 총 batch의 개수입니다.
            num_val_batches (_type_): evaluation 중 mean epoch loss를 연산하기 위해 필요한 총 batch의 개수입니다.
        """
        self.save_dir_name = save_dir_name
        
        self.settings = settings
        self.train_cfg = train_cfg
        self.val_cfg = val_cfg
        
        self.model = model
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_train_batches = num_train_batches
        self.num_val_batches = num_val_batches

        self.accumulation_step = 4


    def base_trainer(self):
        """torchvision.models.segmentation의 모델들을 학습시킬 때 사용하는 trainer입니다.
        """
        
        print(f"Start training..")
        model = self.model.to(self.device)

        best_dice = 0.
        saved_models = deque()
        # training loop
        for epoch in range(self.train_cfg['num_epochs']):
            model.train()

            total_loss = 0.
            for step, (images, masks) in enumerate(self.train_loader):
                # gpu 연산을 위해 device 할당
                images, masks = images.to(self.device), masks.to(self.device)

                # forward
                if self.settings['lib'] == 'smp':
                    outputs = model(images)
                else:
                    outputs = model(images)['out']

                # loss 계산 & update
                loss = self.criterion(outputs, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

                # log_interval에 따라 training 과정 중 연산된 값들을 출력
                if (step + 1) % self.train_cfg['log_interval'] == 0:
                    print(
                        f'Epoch [{epoch + 1}/{self.train_cfg["num_epochs"]}], '
                        f'Step [{step + 1}/{len(self.train_loader)}], '
                        f'Loss: {round(loss.item(), 4)}'
                    )
            
            # wandb 기록 - training
            wandb.log({
                "Epoch": epoch + 1,
                "Train/Mean_Epoch_Loss": round(total_loss / self.num_train_batches, 4),
            })
            
            # evaluation
            save_dir = os.path.join(self.settings['saved_dir'], self.save_dir_name)

            if (epoch + 1) % self.val_cfg['val_every'] == 0:
                dice = validation(self.settings, self.device, epoch + 1,
                                model, self.val_loader, self.criterion, self.num_val_batches) # wandb 기록은 validation 함수 내부에서 진행됨

                self.scheduler.step(dice)

                if best_dice < dice:
                    print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                    best_dice = dice
                    
                    if len(saved_models) >= self.val_cfg['val_save_limit']:
                        # Delete the oldest model
                        oldest_model_path = saved_models.popleft()
                        os.remove(os.path.join(save_dir, oldest_model_path))
                        print(f"{oldest_model_path} is deleted")
                    
                    # Save the current model
                    save_file_name = f"{self.train_cfg['models']}_{epoch+1}_{dice:.4f}.pt"

                    save_model(model, save_dir, save_file_name)
                    saved_models.append(save_file_name)

                    print(f"Model is saved in {os.path.join(save_dir, save_file_name)}")

    # TODO: gradient accumulation trainer 구현
    def gradient_accumulation_trainer(self):
        print(f"Start training with Gradient Accumulation...")
        model = self.model.to(self.device)

        best_dice = 0.
        saved_models = deque()
        # training loop
        for epoch in range(self.train_cfg['num_epochs']):
            model.train()

            total_loss, accumulation_loss = 0., 0.
            self.optimizer.zero_grad()
            for step, (images, masks) in enumerate(self.train_loader):
                # gpu 연산을 위해 device 할당
                images, masks = images.to(self.device), masks.to(self.device)

                # forward
                if self.settings['lib'] == 'smp':
                    outputs = model(images)
                else:
                    outputs = model(images)['out']

                # loss 계산 & normalize
                loss = self.criterion(outputs, masks)
                loss = loss / self.accumulation_step
                loss.backward()

                accumulation_loss += loss.item()

                # accumulation step의 배수일 때만 parameters update & log 출력
                if (step + 1) % self.accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += accumulation_loss
                    
                    print(
                        f'Epoch [{epoch + 1}/{self.train_cfg["num_epochs"]}], '
                        f'Step [{(step + 1) // self.accumulation_step}/{len(self.train_loader) // self.accumulation_step}], '
                        f'Loss: {round(accumulation_loss, 4)}'
                    )

                    # 초기화
                    accumulation_loss = 0.

            self.scheduler.step()
            
            # wandb 기록 - training
            wandb.log({
                "Epoch": epoch + 1,
                "Train/Mean_Epoch_Loss": round(total_loss / (len(self.train_loader) // self.accumulation_step), 4),
            })
            
            # evaluation
            save_dir = os.path.join(self.settings['saved_dir'], self.save_dir_name)

            if (epoch + 1) % self.val_cfg['val_every'] == 0:
                dice = validation(self.settings, self.device, epoch + 1,
                                model, self.val_loader, self.criterion, self.num_val_batches) # wandb 기록은 validation 함수 내부에서 진행됨

                if best_dice < dice:
                    print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                    best_dice = dice
                    
                    if len(saved_models) >= self.val_cfg['val_save_limit']:
                        # Delete the oldest model
                        oldest_model_path = saved_models.popleft()
                        os.remove(os.path.join(save_dir, oldest_model_path))
                        print(f"{oldest_model_path} is deleted")
                    
                    # Save the current model
                    save_file_name = f"{self.train_cfg['models']}_{epoch+1}_{dice:.4f}.pt"

                    save_model(model, save_dir, save_file_name)
                    saved_models.append(save_file_name)

                    print(f"Model is saved in {os.path.join(save_dir, save_file_name)}")