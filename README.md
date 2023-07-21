## **Hand Bone Image Segmentation (Naver Connect Foundation - boostcamp AI Tech CV-02조 팀 멋쟁이)**

### 📌 **대회 정보**
- - -
- **대회 주제** <br>
  뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는데 필수적입니다.
Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며 다양한 목적으로 도움을 줄 수 있습니다.
본 대회에서는 손 뼈를 촬영한 X-Ray 이미지를 입력 받아 **특정 영역 별로 segmentation을 수행하는 모델**을 만들어야 합니다.
- **대회 목표**
    - 순위보단 기초 개념에 집중하여 많은 것을 얻어가기
- **대회 일정** : 23.06.07 10:00 ~ 23.06.22 19:00 (3주)

### 🐦 **Members**
- - -
|**이름**|**역할**|**github**|
|:-:|:-:|:-:|
|김성한|Model backbone 및 Augmentaion 실험, CLAHE 구현 및 관련 코드 수정 후 실험|[Happy-ryan](https://github.com/Happy-ryan)|
|박수영|베이스라인 코드 구현 및 인터페이스 개선 / hparam, backbone, architecture 별 성능 실험|[nstalways](https://github.com/nstalways)|
|이다현|Model backbone 및 Augmentaion 실험, Ensemble 및 TTA 실험|[DaHyeonnn](https://github.com/DaHyeonnn)|
|이채원|backbone model 및 Combine Loss, TTA, K-Fold ensemble 실험|[Chaewon829](https://github.com/Chaewon829)|
|정호찬|Unet계열(Unet, Unet++, Unet3+) 구현 및 실험, Segformer 구현 및 실험, Loss function 구현 및 실험, inference시 시각화 해주는 기능 구현|[Eumgil98](https://github.com/Eumgill98)|

### 📝 **Dataset 개요**
- - -
- Image
    - **손 뼈를 촬영한 X-Ray 이미지**
    - **2048 x 2048 x 3**
- Annotation
    - X-Ray 이미지에 대한 segmentation annotation (type: json)
    - 크게 손가락/손등/팔로 구성되며, 세부적으로는 **총 29개의 클래스가 존재**함
- 전체 이미지 개수
    - Train 800 장
    - Test 300 장

### 🐤 **폴더 구조**
- - -
```
📦level2_cv_semanticsegmentation-cv-02
 ┣ 📂code
 ┃ ┣ 📜evaluation.py
 ┃ ┣ 📜inference.py
 ┃ ┣ 📜metrics.py
 ┃ ┣ 📜my_augmentations.py
 ┃ ┣ 📜my_criterion.py
 ┃ ┣ 📜my_dataset.py
 ┃ ┣ 📜my_models.py
 ┃ ┣ 📜my_trainer.py
 ┃ ┣ 📜train.py
 ┃ ┗ 📜utils.py
 ┗ 📂configs
 ┃ ┗ 📜baseline.yaml
```

### 🐧 **최종 결과**
- - -
```
🏅Public score : 7 / 19
🏅Private score :  10 / 19
```
