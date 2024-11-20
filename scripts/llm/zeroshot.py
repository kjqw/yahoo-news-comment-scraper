# %%
import torch
from transformers import pipeline

# %%
# 使用するデバイス
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# モデルと入力データ
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
texts = [
    "新しいiPhoneの価格が高すぎると批判されています。",
    "日本代表がワールドカップで優勝しました！",
    "映画のレビューが酷評されていました。",
    "新しいスマートフォンが発売されました。",
]
labels_category = ["スマートフォン", "エンタメ", "スポーツ"]
labels_sentiment = ["ポジティブ", "中立", "ネガティブ"]

# プロンプトテンプレート
hypothesis_template_category = "この文章は{}に関する内容です。"
hypothesis_template_sentiment = "この文章の感情は{}です。"

# %%
try:
    # モデルロード
    zeroshot_classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=0 if device.type == "cuda" else -1,
    )

    # ゼロショット分類
    results_category = zeroshot_classifier(
        texts,
        labels_category,
        hypothesis_template=hypothesis_template_category,
        # multi_label=False,
    )
    results_sentiment = zeroshot_classifier(
        texts,
        labels_sentiment,
        hypothesis_template=hypothesis_template_sentiment,
        multi_label=False,
    )

except Exception as e:
    print(f"エラーが発生しました: {e}")

# %%
# メモリ解放
del zeroshot_classifier
torch.cuda.empty_cache()

# %%
results_category

# %%
results_sentiment

# %%
