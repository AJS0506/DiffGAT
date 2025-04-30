import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda:0"  # 혹은 'cpu'

model_name = "bert-base-uncased"

# 1) 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)
model.eval()

# 2) 예시 문장
temp = ["테스트용 단어입니다." for _ in range(10)]

# 3) 토크나이징
inputs = tokenizer(
    temp,
    return_tensors="pt",
    padding=True,
    truncation=True
)

# 4) GPU 등 디바이스로 이동
inputs = {k: v.to(device) for k, v in inputs.items()}

# 5) 모델 수행 (gradient 추적 X)
with torch.no_grad():
    outputs = model(**inputs)
    # 일반적으로 BERT의 마지막 레이어 히든스테이트는 outputs.last_hidden_state에 있음
    last_hidden_state = outputs.last_hidden_state

# 예) [CLS] 토큰 임베딩만 추출하고 싶다면
cls_embeddings = last_hidden_state[:, 0, :]  # batch x hidden_dim

print("전체 last_hidden_state 모양 :", last_hidden_state.shape)
print("CLS 임베딩 모양 :", cls_embeddings.shape)
