import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# GPU が利用可能か確認
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Hugging Face Hub 上のリポジトリからモデルとトークナイザーをロード
model_name = "ayousanz/modernbert-vtuber-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)  # GPU が利用可能なら GPU に移動

# 推論用パイプラインの作成
vtuber_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)

# 5つのサンプルテキストで推論例を実行
sample_texts = [
    "この動画では、バーチャルなキャラクターがリアルタイムに動く様子を配信しています。",
    "このチャンネルは、料理レシピの動画を投稿しています。",
    "最新のVTuberがライブ配信を行っており、視聴者との交流が盛んです。",
    "旅行動画を中心に、世界各地の観光地を紹介しています。",
    "ここでは、3Dモデルを使ったアニメーション動画を配信しています。"
]

for text in sample_texts:
    result = vtuber_classifier(text)
    print("入力:", text)
    print("推論結果:", result)
    print("-" * 50)

