import torch
import random
import copy
import sys
import os
import datetime

from collections import defaultdict
from pympler import asizeof

# ============= 모델 임포트 =============
from Model.MyGAT import DiffHeadGATRating

from GraphMaker.make_graph import GraphMaker
from DataLoader.load_dataset import DataLoader

from Eval.result import calcEvaluationScore
from Eval.tracker import TrainTracker


# ============= 시드 고정 =============
def set_random_seed(seed):
    random.seed(seed)                # Python random 시드
    # np.random.seed(seed)             # NumPy 시드
    torch.manual_seed(seed)          # PyTorch CPU 시드
    torch.cuda.manual_seed(seed)     # PyTorch GPU 시드 (Single GPU)
    torch.cuda.manual_seed_all(seed) # Multi-GPU도 쓰는 경우
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

RANDOM_SEED = 1008
set_random_seed(RANDOM_SEED)

# 4대의 GPU머신에서 테스트했습니다.
# 편의상 병렬 학습하기 위해 GPU설정 탭에서 f"cuda:{DATASET_NUMBER}" 로 GPU가 설정됩니다. 
DATASET_NUMBER = 4

# ============= 데이터셋 정의 =============
dataset_list = ["gowalla", "movielens_small", "movielens_25M", "netflixPrize", "filmtrust"]
DATA_SET = dataset_list[DATASET_NUMBER]

# ============= 학습, 검증, 테스트 데이터셋 로드 =============
loader = DataLoader(DATA_SET)
train_set, val_set, test_set = loader.load_dataset()

# 메모리 사용량 계산
size_bytes = asizeof.asizeof(train_set)      # train_set과 모든 하위 객체 포함
size_mb    = size_bytes / (1024 ** 2)        # Byte → MB(2^20)

print(f"train_set 메모리 사용량: {size_mb:.2f} MB ({size_bytes:,} bytes)")
print("시드 -> ", RANDOM_SEED)
print("데이터셋 -> ", DATA_SET)
# ============= 데이터셋으로 그래프 만들기 =============
gm = GraphMaker()
graph = gm.get_graph(dataset=DATA_SET)
num_type1_nodes, num_type2_nodes = gm.get_num_nodes()

print("학습 그래프 -> ",graph)

# ============= TimeStamp Z‑정규화 =============

# 1) train_set에서 timestamp만 모아 float32 텐서로
ts_train = torch.tensor(
    [float(ts) for _, _, ts, _ in train_set],       # ← 반드시 float!
    dtype=torch.float32
)

# 2) 평균 (μ)·표준편차 (σ) 계산
μ = ts_train.mean()
σ = ts_train.std(unbiased=False).clamp_min(1e-8)   # 0 나눗셈 방지

def z_normalize(dataset, μ, σ):
    return [
        (src, dst, (float(ts) - μ.item()) / σ.item(), rt)
        for src, dst, ts, rt in dataset
    ]

# 3) train/val/test 모두 같은 Z‑스케일러 적용
train_set = z_normalize(train_set, μ, σ)
val_set   = z_normalize(val_set,   μ, σ)
test_set  = z_normalize(test_set,  μ, σ)

# ==============  노드별 Degree 데이터 처리  ===========
uid2dg = defaultdict(int)
mid2dg = defaultdict(int)
for uid, mid, ts, rating in train_set:
    uid2dg[uid] += 1
    mid2dg[mid] += 1 

# ============== 유저별 TimeStamp 데이터 처리 ==========
""" Key -> user ID, Val -> timestamp"""
uid2ts = defaultdict(list)
for uid, mid, ts, rating in train_set:
    uid2ts[uid].append(ts)






# ============== 유저별 Rating 데이터 처리 ==========
""" Key -> user ID, Val -> Ratings"""
uid2rt = defaultdict(list)
for uid, mid, ts, rating in train_set:
    uid2rt[uid].append(float(rating))

# ============== 아이템(영화)별 Rating 데이터 처리 ==========
""" Key -> item ID, Val -> Ratings"""
mid2rt = defaultdict(list)
for uid, mid, ts, rating in train_set:  
    mid2rt[mid].append(float(rating))



# ============= 학습 모델 초기화 및 GPU 설정 =============
device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
graph = graph.to(device)
embedding_dim = 128
first_layer_dim = 64
second_layer_dim = 32

model = DiffHeadGATRating(num_type1_nodes, num_type2_nodes, embedding_dim, first_layer_dim, second_layer_dim, uid2ts, uid2dg, mid2dg, uid2rt, mid2rt, device).to(device)


# ============= 옵티마이저 설정 =============
LEARNING_RATE = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)


# ============= Hard Negative 샘플링을 위한 데이터셋 딕셔너리 =============
user_visited = defaultdict(set)
for uid, mid, _, _ in train_set + val_set + test_set:
    user_visited[uid].add(mid)


# ============= 에포크 및 배치사이즈 설정 =============
epoch = 10**9
best_batch_dict = {"gowalla": 126843, "movielens_25M": 497893}
batch_size = len(train_set) // 30
print(f"훈련 세트 개수 -> {len(train_set)}, batch_size -> {batch_size}")


# ============= Loss Tracker =============
train_loss_tracker = []
val_loss_tracker = []


# ============= Early Stopping 구현 =============
early_stop_cnt = 0
STOP_CONDITION = 10 # 얼마나 참을 것 인지!

best_val_loss = float('inf')
best_model_state = None









# ============= 학습 시작! =============
for e in range(epoch):
    # ============================
    # 1) Training
    # ============================
    model.train()
    random.shuffle(train_set)  # 매 에폭마다 학습 데이터 순서를 무작위로 섞기
    num_samples = len(train_set)
    total_train_loss = 0.0

    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_data = train_set[i:end_idx]

        pos_users = torch.LongTensor([row[0] for row in batch_data]).to(device)
        pos_items = torch.LongTensor([row[1] for row in batch_data]).to(device)

        # GCN 호출
        h = model(graph)
        user_emb = h['user']
        item_emb = h['item']

        # =========== Random Hard Negative Sampling ===========
        neg_items_list = []
        pos_users_cpu = pos_users.cpu().tolist()  # 유저 ID를 CPU 리스트로 변환 (set/dict 접근 속도↑)

        for u in pos_users_cpu:
            visited_set = user_visited[u] # 이 유저가 이미 본 아이템 집합
            while True:
                candidate = random.randint(0, num_type2_nodes - 1)
                if candidate not in visited_set:
                    neg_items_list.append(candidate)
                    break
        
        neg_items = torch.tensor(neg_items_list, dtype=torch.long, device=device)

        # BPR loss 계산
        pos_scores = (user_emb[pos_users] * item_emb[pos_items]).sum(dim=1)
        neg_scores = (user_emb[pos_users] * item_emb[neg_items]).sum(dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        optimizer.zero_grad()
        bpr_loss.backward()
        optimizer.step()

        total_train_loss += bpr_loss.item()

    avg_train_loss = total_train_loss / (num_samples // batch_size + 1)

    train_loss_tracker.append(avg_train_loss)

    # ============================
    # 2) Validation
    # ============================
    model.eval()
    total_val_loss = 0.0
    val_samples = len(val_set)

    with torch.no_grad():
        for i in range(0, val_samples, batch_size):
            end_idx = min(i + batch_size, val_samples)
            val_batch = val_set[i:end_idx]

            pos_users = torch.LongTensor([row[0] for row in val_batch]).to(device)
            pos_items = torch.LongTensor([row[1] for row in val_batch]).to(device)

            # 모델 순전파
            h = model(graph)
            user_emb = h['user']
            item_emb = h['item']

            # 학습시와 동일한 하드 네거티브 샘플링!
            neg_items_list = []
            pos_users_cpu = pos_users.cpu().tolist()

            for u in pos_users_cpu:
                visited_set = user_visited[u]
                while True:
                    candidate = random.randint(0, num_type2_nodes - 1)
                    if candidate not in visited_set:
                        neg_items_list.append(candidate)
                        break

            neg_items = torch.tensor(neg_items_list, dtype=torch.long, device=device)

            pos_scores = (user_emb[pos_users] * item_emb[pos_items]).sum(dim=1)
            neg_scores = (user_emb[pos_users] * item_emb[neg_items]).sum(dim=1)
            val_bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            total_val_loss += val_bpr_loss.item()

    avg_val_loss = total_val_loss / (val_samples // batch_size + 1)

    val_loss_tracker.append(avg_val_loss)

    print(f"[Epoch {e+1}/inf] "
          f"Train Loss = {avg_train_loss:.4f}, "
          f"Val Loss = {avg_val_loss:.4f}")

    # ============ Early Stopping 관련 코드 ================
    
    if e < 10:
        print("초기 학습.. (early stopping skip!)")
        continue

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"  >> Best model updated at epoch {e+1}, Val_loss = {avg_val_loss:.4f}")
        early_stop_cnt = 0  # 개선되었으므로 카운트 리셋
    else:
        early_stop_cnt += 1
        print("early stop cnt ->", early_stop_cnt)

    if early_stop_cnt >= STOP_CONDITION:
        print(f"early stop cnt reached {STOP_CONDITION}, stop")
        break




# ============ 가장 좋은 성능 불러오기 및 테스트 시작 ================

# Best 모델 state를 파일로 저장
model_class_name = model.__class__.__name__
time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = f"BestModel/{RANDOM_SEED}_{LEARNING_RATE}_{model_class_name}_{DATA_SET}_{batch_size}_{STOP_CONDITION}_{embedding_dim}_{first_layer_dim}_{second_layer_dim}_{time_stamp}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Best model parameters have been loaded.")

    torch.save(best_model_state, f"{folder_path}/best_Model.pth")
    print("최적 모델 저장 완료!")


model.eval()
with torch.no_grad():
    test_hidden = model(graph)

# test_processor = calcEvaluationScore(test_set, test_hidden, folder_path)
# avg_recall, avg_precision, avg_ndcg = test_processor.calcScore()

# tracker = TrainTracker(train_loss_tracker, val_loss_tracker, avg_recall, avg_precision, avg_ndcg, folder_path)
# tracker.plot_results()

test_processor = calcEvaluationScore(test_set, test_hidden, folder_path, DATA_SET)
(macro_recall, macro_precision, macro_ndcg,
 micro_recall, micro_precision, micro_ndcg) = test_processor.calcScore()

# 2) 트래커 ㄱㄱㄱ
tracker = TrainTracker(
    train_loss_tracker, 
    val_loss_tracker,
    macro_recall, macro_precision, macro_ndcg,
    micro_recall, micro_precision, micro_ndcg,
    folder_path
)

# 3) 결과 플롯 및 저장
tracker.plot_results()



