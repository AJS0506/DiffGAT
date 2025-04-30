# DFGAT

본 프로젝트는 KCC 2025 정보과학회 학술대회에 제출된 사용자 추천 시스템을 위한 그래프 기반 딥러닝 모델 DFGAT의 구현 코드입니다.  
DFGAT 모델은 시간 정보와 인기도 정보를 활용한 다중 헤드 그래프 어텐션 네트워크를 사용하여 추천 성능을 향상시키는 연구 결과입니다.  

## 디렉토리 구조

```
2025_KCC_GIT/
├── DFGAT/                 # 베이스라인 모델 성능평가를 위한 폴더
│   ├── 베이스라인/         # 베이스라인 모델 (GCN, GAT, SAGE, GINC)
│   │   ├── baseline_train.py  # 베이스라인 모델 학습 스크립트
│   │   ├── BestModel/      # 학습된 최적 모델 저장 디렉토리
│   │   ├── Data/           # 데이터셋 디렉토리 **(다운로드 받은 데이터가 위치는 폴더 입니다.)**
│   │   ├── DataLoader/     # 데이터 로딩 클래스
│   │   ├── Eval/           # 평가 지표 계산 및 결과 추적
│   │   ├── GraphMaker/     # 그래프 생성 클래스
│   │   ├── Model/          # 모델 구현 클래스
│   │   └── 실행방법.txt    # 베이스라인 모델 실행 가이드
│   └── 데이터셋_전처리/    # 데이터셋 전처리 관련 코드
│
├── DFGAT_addRating/        # 제안하는 DFGAT 학습 폴더
│   ├── MyGAT/              # 평점 기반 DFGAT 모델 구현
│   │   ├── BestModel/      # 학습된 최적 모델 저장 디렉토리
│   │   ├── Data/           # 데이터셋 디렉토리 **(다운로드 받은 데이터가 위치는 폴더 입니다.)**
│   │   ├── DataLoader/     # 데이터 로딩 클래스
│   │   ├── Eval/           # 평가 지표 계산 및 결과 추적
│   │   ├── GraphMaker/     # 그래프 생성 클래스
│   │   ├── Model/          # 향상된 모델 구현 클래스 (평점 정보 포함)
│   │   ├── train.py        # Movielens-small, Movielens-25M 데이터셋 학습 코드
│   │   ├── train_gowalla.py # Gowalla 데이터셋 학습 -> 추가 연구 진행 중
│   │   └── 절대경로 위치.txt # 절대 경로 설정 안내
│   └── 데이터셋_전처리/    # 데이터셋 전처리 관련 코드
|       ├── FilmTrust/      # FilmTrust 데이터셋
│       ├── Gowalla/        # Gowalla 데이터셋
│       ├── MovieLens-25M/  # MovieLens-25M 데이터셋
│       ├── MovieLens-small/ # MovieLens-small 데이터셋
│       ├── Netflix-prize/  # Netflix-prize 데이터셋
│       └── 최종 데이터셋 형식 # 데이터셋 형식 설명
│
└── 성능평가_V7.xlsx        # 모델 성능 평가 결과
```

## 실행 방법

### 1. 데이터셋 준비

데이터셋은 (https://drive.google.com/file/d/1sdcXQILxplVsvEtui_emW9tccZu9L-id/view?usp=drive_link)에서 다운로드 받을 수 있습니다.   
다운로드 후 각 모델의 Data 폴더에 배치하면 됩니다.  

학습 데이터셋은 다음과 같이 구성되어 있습니다.  
[user_id, item_id, timestamp, rating, ...]   
유저와 아이템에 대한 상호작용 데이터이며 학습데이터 : 검증데이터 : 테스트데이터 를 6:2:2로 분할하였습니다.  
유저별 timestamp를 정렬하여, 실제 추천시스템 시나리오로 가정하였습니다.  
(현재 데이터로 학습 후, 미래 예측)  


### 2. 경로 설정

사용 환경에 맞게 절대 경로를 설정해야 합니다. 수정이 필요한 파일은 다음과 같습니다:
- `DFGAT_addRating/MyGAT/GraphMaker/make_graph.py`: 23~27라인
- `DFGAT_addRating/MyGAT/DataLoader/load_dataset.py`: 10~12라인

### 3. 베이스라인 모델 학습 코드 실행

```bash

# MovieLens-small 데이터셋 실행 예시
python3 baseline_train.py --gpu 0 --dataset 1 --seed 1004 --model GCN
python3 baseline_train.py --gpu 1 --dataset 1 --seed 1004 --model GAT
python3 baseline_train.py --gpu 2 --dataset 1 --seed 1004 --model SAGE
python3 baseline_train.py --gpu 3 --dataset 1 --seed 1004 --model GINC

# MovieLens-25M 데이터셋 실행 예시
python3 baseline_train.py --gpu 0 --dataset 2 --seed 1004 --model GCN
python3 baseline_train.py --gpu 1 --dataset 2 --seed 1004 --model GAT
python3 baseline_train.py --gpu 2 --dataset 2 --seed 1004 --model SAGE
python3 baseline_train.py --gpu 3 --dataset 2 --seed 1004 --model GINC

# FilmTrust 데이터셋 실행 예시
python3 baseline_train.py --gpu 0 --dataset 5 --seed 1004 --model GCN
python3 baseline_train.py --gpu 1 --dataset 5 --seed 1004 --model GAT
python3 baseline_train.py --gpu 2 --dataset 5 --seed 1004 --model SAGE
python3 baseline_train.py --gpu 3 --dataset 5 --seed 1004 --model GINC
```

### 4. DFGAT 학습 코드 실행

```bash
# movielens-small, movielens-25M, FilmTrust 데이터셋 실행 (GPU 및 SEED는 하드코딩 되어 있습니다)
python3 train.py
```

## 모델 설명

### DFGAT 모델

## 1. 그래프 구조 정의
- **노드**: 사용자(User)와 아이템(Item)
- **엣지**: 사용자와 아이템 간의 상호작용 관계

## 2. 노드 특징 표현
- 각 사용자 및 아이템 노드의 초기 특징 벡터 $h$는 일반화를 위해 $d$ 차원의 학습 가능한 파라미터로 정의

## 3. Multi-head MLP 구조
- 유저와 아이템의 특징 벡터를 Multi-head MLP의 입력으로 사용
- 각 노드의 특징을 헤드 개수만큼 확장

## 4. 어텐션 메커니즘 (3개 헤드)
각 헤드는 서로 다른 데이터 적응적 어텐션 연산 수행:

1. **첫 번째 헤드**: 기존의 GATConv 연산
   - 그래프 어텐션 네트워크 기반 정보 집계

2. **두 번째 헤드**: RatingConv 연산
   - 사용자-아이템 간 평점 정보를 고려한 어텐션 계산

3. **세 번째 헤드**: PopConv 연산
   - 노드의 차수(연결 수)를 고려한 인기도 기반 어텐션 계산

## 5. 정보 통합
- 세 개 헤드의 출력을 concatenate 연산으로 결합
- 결합된 특징에 MLP를 적용하여 최종 표현 학습
- 결과적으로 사용자와 아이템에 대한 학습된 표현 행렬 획득

## 6. 최적화
- 학습된 표현 행렬을 이용하여 BPR(Bayesian Personalized Ranking) 최적화 수행
- 사용자의 상호작용 학습 데이터를 활용하여 개인화된 순위 모델 학습

## 전체 프로세스 요약
사용자-아이템 그래프 → 초기 특징 표현 → Multi-head MLP → 3가지 어텐션 연산 → 정보 통합 → 최종 표현 행렬 → BPR 최적화


<img width="510" alt="image" src="https://github.com/user-attachments/assets/9f5afe22-3b2f-4d02-942b-c2e86a517930" />


MovieLens 데이터셋 평가 시에는 GATConv, RatingConv, PopConv 3개를 사용하였습니다.

해당 프로젝트의 코드에는 timestamp 정보를 반영하는 TimeConv도 구현되어있습니다.

## 결과 및 평가

학습된 모델의 평가 결과는 `성능평가_V6.xlsx` 파일에서 확인할 수 있습니다.  
평가 지표로는 Recall, Precision, NDCG 등을 사용하였습니다.  
학습 과정에서 생성된 모델과 평가 결과는 각 모델의 BestModel 디렉토리에 저장됩니다.  


<img width="618" alt="image" src="https://github.com/user-attachments/assets/4a14217e-a680-47bd-b8f0-c82c84b6da09" />



DFGAT는 Movielens-small, Movielens-25M 데이터셋에서   
Recall@20, Precision@20, NDCG@20 모두 베이스라인보다 우수한 성능을 달성하였습니다.    
Recall, Precision 지표의 경우 약간의 성능 향상이 있었으나 오차범위 이내로 측정되었지만 NDCG 지표의 경우 유의미한 성능 향상을 관측하였습니다.    

## 참고사항

- 학습 환경은 ubuntu 22.04 버전에서 수행되었습니다.  
- CUDA 11.8 / Pytorch 2.4.1 / CUDA와 pytorch 버전에 맞는 DGL 공식 홈페이지의 pip 설치 명령어를 사용하였습니다.  
- 학습 속도 향상을 위해 여러 GPU를 사용할 수 있도록 설계되었습니다. Movielens-25M 데이터셋의 경우 1회 학습시 10GB 정도의 VRAM이 소요됩니다.  
- 성능 평가는 RTX A6000 48GB X 4, RAM 256, CPU 32Core 환경에서 수행되었습니다.  
