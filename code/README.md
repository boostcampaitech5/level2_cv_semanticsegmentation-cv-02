## Baseline 코드 사용법 (2023-06-09 updated)
### Settings
1. `git clone ~` 명령어를 사용하여 repository를 복제합니다.
2. 기본으로 제공되는 data를 `level2_cv_semanticsegmentation-cv-02/` 아래로 복사합니다.
```
# 2번까지 수행했을 때 디렉토리 구조

level2_cv_semanticsegmentation-cv-02
├── .git
├── .github
├── code
│   ├── evaluation.py : 학습한 모델의 성능을 평가할 때 사용하는 함수들이 정의되어 있습니다.
│   ├── inference.py : submission을 위한 csv 파일을 만들 때 사용합니다.
│   ├── metrics.py : 성능 평가 시 사용하는 metric들이 정의되어 있습니다.
│   ├── my_dataset.py : 학습/평가에 사용할 데이터들을 구조화하는 dataset class가 정의되어 있습니다.
│   ├── my_models.py : 모델 종류들이 정의되어 있습니다.
│   ├── my_trainer.py : 모델을 학습시킬 때 사용하는 train 함수들이 정의되어 있습니다.
│   ├── train.py : 모델 학습 시 사용하는 모듈입니다.
│   └── utils.py : 실험에 필요한 각종 함수들을 정의해 둔 모듈입니다.
│
├── configs : 실험에 사용할 yaml 파일들을 모아놓은 폴더입니다.
|   ├── baseline.yaml : 제공받은 코드의 초기 세팅들을 모아놓은 yaml 파일입니다.
|   |
|   └── sy : 개인 yaml 폴더. 본인 이름의 이니셜로 폴더를 세팅하시면 됩니다.
|       └── 01_fcn_resnet50_bs4.yaml : 실험에 사용할 yaml 파일입니다.
|
├── data : 데이터들을 모아둔 폴더입니다. (push가 **불가능**하고, .gitignore에 추가되어 있습니다)
|
├── submissions : inference.py를 통해 생성한 csv 파일이 저장됩니다. (infernece.py를 실행하면 자동으로 디렉토리가 만들어집니다)
|
├── trained_models : 실험에 사용할 yaml 파일들을 모아놓은 폴더입니다. (push는 가능하지만, .gitignore에 추가되어 있습니다)
│
├── .gitignore : commit하지 않을 폴더, 파일들을 기록해두는 곳입니다.
├── .gitmessage.txt : commit template 입니다. 사용법은 아래에서 설명드리겠습니다.
└── README.md
```
3. `configs` 폴더 아래에 본인 이름의 이니셜로 폴더를 생성하고, 그 아래에 yaml 파일을 만듭니다.<br>
예를 들자면, `configs/sy/01_fcn_resnet50_bs4.yaml` 처럼 만들어주시면 됩니다.<br>
yaml 파일의 구조는 다음과 같습니다.
```yaml
# (2023-06-09 updated)
settings: # 실험을 위해 기본적으로 세팅하는 값들입니다.
  # data path (학습에 사용할 데이터의 경로를 정의합니다)
  # 위에서 얘기한 방식대로 디렉토리를 구성했을 경우, 경로는 그대로 사용하시면 됩니다.
  image_root: "../data/train/DCM"
  label_root: "../data/train/outputs_json"
  tt_image_root: "../data/test/DCM"

  # save path (학습한 모델을 저장할 경로입니다.)
  saved_dir: "../trained_models"
  submission_dir: "../submissions"

  # seed (실험의 재현 가능성을 위해 설정하는 시드 값입니다.)
  seed: 21

  # classes of hand bone (손 뼈의 클래스들을 정의해둔 list입니다.)
  classes: [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna']

train: # train에 사용할 값들입니다.
  models: 'fcn_resnet50' # 제일 중요한 부분으로, my_models.py에 작성된 docstring을 꼭 읽어주세요.
  num_epochs: 40 # 총 에폭입니다.
  batch_size: 8 # train 데이터의 batch size입니다.
  shuffle: True # train 데이터로더의 shuffle 인자에 들어갈 값입니다.
  num_workers: 8 # train 데이터로더의 num_workers 인자에 들어갈 값입니다.
  drop_last: True # train 데이터로더의 drop_last 인자에 들어갈 값입니다.
  lr: 0.0001 # optimizer의 learning rate입니다.
  log_interval: 25 # log_interval 값에 따라 train loss의 터미널 출력 주기가 결정됩니다.
  weight_decay: 0.00001 # optimizer의 weight_decay 인자에 들어갈 값입니다.

val: # validation에 사용할 값들입니다.
  batch_size: 2
  shuffle: False
  num_workers: 2
  drop_last: False
  val_every: 1 # val_every 값에 따라 evaluation 주기가 결정됩니다.

test: # inference에 사용할 값들입니다.
  batch_size: 2
  shuffle: False
  num_workers: 2
  drop_last: False

```
## Training
- 세팅을 완료하고, yaml 파일을 생성하셨다면 training이 가능합니다. Training 방법은 간단합니다.
- 터미널에 `python train.py --config_path ../configs/your_folder/your_yaml_file.yaml` 을 입력하고, 실행하시면 됩니다.
- 학습 과정은 "Hand Bone Segmentation'이라는 프로젝트 이름으로 WandB에 기록됩니다.
- **2023-06-09** 기준 **train 과정에서 epoch, mean epoch loss**가 기록되며,<br>
**valid 과정에서 mean epoch loss, average dice coefficients와 finger-16, Trapezoid, Pisiform에 대한 dice coefficients**를 확인하실 수 있습니다.
- Epoch을 기록하는 이유는, WandB의 그래프 x축은 기본 step 수로 세팅되어 있습니다. 이를 epoch 별로 확인하기 위해 기록합니다.<br>
WandB 상의 그래프 x축을 변경하는 방법은, 그래프 우상단에 있는 edit 아이콘을 누르시고 x축을 epoch으로 변경해주시면 됩니다.
- - -
- 학습한 모델은 `../trained_models` 아래에 저장됩니다. 모델 저장은 validation set에 대한 average dice coefficients 값이 갱신될 때마다 이루어집니다.
```
# (2023-06-09 updated)
trained_models
└── 01_fcn_resnet50_bs4_best.pth # inference에 사용한 yaml 파일의 이름을 바탕으로 pth 파일이 저장됩니다.
```
- - -
## Inference
- 학습이 완료된 모델과, test data를 기반으로 inference가 가능합니다. 사용법은 다음과 같습니다.
```bash
# Example
python inference.py --config_path ../configs/your_folder/your_yaml_file.yaml --model_path ./trained_models/something_best.pth
```
- 실행 결과는 inference.py 실행에 사용한 yaml 파일의 이름을 기반으로 저장됩니다.
```
submissions
└── your_yaml_file.csv
```
## Visualization
- 업데이트 예정입니다.
## Others
- 코드를 사용하시다가 추가하고 싶은 기능이 있으시다면 PR을, 버그가 있다면 issue를 활용해주세요! 감사합니다 🙇