# DFGAT (Differential Feature Graph Attention Network)

본 프로젝트는 사용자 추천 시스템을 위한 그래프 기반 딥러닝 모델 DFGAT의 구현 코드입니다. DFGAT 모델은 시간 정보와 인기도 정보를 활용한 다중 헤드 그래프 어텐션 네트워크를 사용하여 추천 성능을 향상시키는 연구 결과입니다.

## 디렉토리 구조

```
2025_KCC_GIT/
├── DFGAT/                  # 기본 DFGAT 모델 구현
│   ├── 베이스라인/         # 베이스라인 모델 (GCN, GAT, SAGE, GINC)
│   │   ├── baseline_train.py  # 베이스라인 모델 학습 스크립트
│   │   ├── BestModel/      # 학습된 최적 모델 저장 디렉토리
│   │   ├── Data/           # 데이터셋 디렉토리
│   │   ├── DataLoader/     # 데이터 로딩 클래스
│   │   ├── Eval/           # 평가 지표 계산 및 결과 추적
│   │   ├── GraphMaker/     # 그래프 생성 클래스
│   │   ├── Model/          # 모델 구현 클래스
│   │   └── 실행방법.txt    # 베이스라인 모델 실행 가이드
│   └── 데이터셋_전처리/    # 데이터셋 전처리 관련 코드
│
├── DFGAT_addRating/        # 평점 정보를 추가한 확장 DFGAT 모델
│   ├── MyGAT/              # 평점 기반 DFGAT 모델 구현
│   │   ├── BestModel/      # 학습된 최적 모델 저장 디렉토리
│   │   ├── Data/           # 데이터셋 디렉토리
│   │   ├── DataLoader/     # 데이터 로딩 클래스
│   │   ├── Eval/           # 평가 지표 계산 및 결과 추적
│   │   ├── GraphMaker/     # 그래프 생성 클래스
│   │   ├── Model/          # 향상된 모델 구현 클래스 (평점 정보 포함)
│   │   ├── train.py        # Netflix 데이터셋 학습 스크립트
│   │   ├── train_gowalla.py # Gowalla 데이터셋 학습 스크립트
│   │   └── 절대경로 위치.txt # 절대 경로 설정 안내
│   └── 데이터셋_전처리/    # 데이터셋 전처리 관련 코드
│       ├── Gowalla/        # Gowalla 데이터셋
│       ├── MovieLens-25M/  # MovieLens-25M 데이터셋
│       ├── MovieLens-small/ # MovieLens-small 데이터셋
│       ├── Netflix-prize/  # Netflix-prize 데이터셋
│       └── 최종 데이터셋 형식 # 데이터셋 형식 설명
│
└── 성능평가_V4.xlsx        # 모델 성능 평가 결과
```

## 실행 방법

### 1. 데이터셋 준비

데이터셋은 [DRIVE_URL]에서 다운로드 받을 수 있습니다. 다운로드 후 각 모델의 Data 폴더에 배치하세요.

### 2. 경로 설정

사용 환경에 맞게 절대 경로를 설정해야 합니다. 수정이 필요한 파일은 다음과 같습니다:
- `DFGAT_addRating/MyGAT/GraphMaker/make_graph.py`: 23~27라인
- `DFGAT_addRating/MyGAT/DataLoader/load_dataset.py`: 10~12라인

### 3. 베이스라인 모델 실행

```bash
# Gowalla 데이터셋 실행 예시
python3 baseline_train.py --gpu 0 --dataset 0 --seed 1005 --model GCN
python3 baseline_train.py --gpu 0 --dataset 0 --seed 1005 --model GAT
python3 baseline_train.py --gpu 0 --dataset 0 --seed 1005 --model SAGE
python3 baseline_train.py --gpu 0 --dataset 0 --seed 1005 --model GINC

# MovieLens-small 데이터셋 실행 예시
python3 baseline_train.py --gpu 0 --dataset 1 --seed 1005 --model GCN
python3 baseline_train.py --gpu 0 --dataset 1 --seed 1005 --model GAT
python3 baseline_train.py --gpu 0 --dataset 1 --seed 1005 --model SAGE
python3 baseline_train.py --gpu 0 --dataset 1 --seed 1005 --model GINC

# Netflix Prize 데이터셋 실행 예시
python3 baseline_train.py --gpu 0 --dataset 3 --seed 1004 --model GCN
python3 baseline_train.py --gpu 1 --dataset 3 --seed 1004 --model GAT
python3 baseline_train.py --gpu 2 --dataset 3 --seed 1004 --model SAGE
python3 baseline_train.py --gpu 3 --dataset 3 --seed 1004 --model GINC
```

### 4. 향상된 DFGAT 모델 실행

```bash
# Netflix Prize 데이터셋 실행
python3 train.py

# Gowalla 데이터셋 실행
python3 train_gowalla.py
```

## 모델 설명

### DFGAT 모델
DFGAT(Differential Feature Graph Attention Network)는 사용자-아이템 상호작용을 이종 그래프로 모델링한 그래프 기반 추천 시스템입니다. 다음과 같은 특징을 가집니다:

1. **다중 헤드 GAT**: 기본 GAT, 시간 정보 인코딩 GAT, 인기도 인코딩 GAT 등 다양한 특성을 학습하는 여러 헤드를 사용

2. **시간 정보 활용**: 사용자의 타임스탬프 데이터를 정규화하여 시간적 패턴을 학습

3. **인기도 정보 활용**: 사용자와 아이템의 연결 정도(degree)를 고려하여 인기도 정보 활용

### DFGAT_addRating 모델
기본 DFGAT 모델에서 평점 정보를 추가로 활용한 확장 모델입니다. RatingEncodingGATConv를 통해 사용자-아이템 간 평점 정보를 어텐션 메커니즘에 반영하여 추천 성능을 향상시켰습니다.

## 결과 및 평가

학습된 모델의 평가 결과는 `성능평가_V4.xlsx` 파일에서 확인할 수 있습니다. 평가 지표로는 Recall, Precision, NDCG 등을 사용하였습니다.

학습 과정에서 생성된 모델과 평가 결과는 각 모델의 BestModel 디렉토리에 저장됩니다.

## 참고사항

- 본 코드는 CUDA가 설치된 환경에서 실행하는 것을 권장합니다.
- 학습 속도 향상을 위해 여러 GPU를 사용할 수 있도록 설계되었습니다.
- 데이터셋 크기에 따라 메모리 사용량이 크게 달라질 수 있으므로 충분한 RAM이 필요합니다.
