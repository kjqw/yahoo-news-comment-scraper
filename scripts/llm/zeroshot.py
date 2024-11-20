# %%
import torch
from transformers import pipeline

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
# model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# text = "Angela Merkel is a politician in Germany and leader of the CDU"
# labels = ["politics", "economy", "entertainment", "environment"]
# hypothesis_template = "This text is about {}"

model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
text = ["今日、新しいiPhoneが発売されました"]
labels = ["スマートフォン", "エンタメ", "スポーツ"]
hypothesis_template = "このニュースは{}に関する文章です."

# %%
zeroshot_classifier = pipeline(
    "zero-shot-classification",
    model=model_name,
    device=device,
)
output = zeroshot_classifier(
    text, labels, hypothesis_template=hypothesis_template, multi_label=False
)

# %%
print(output)

# %%
# メモリ解放
del zeroshot_classifier
torch.cuda.empty_cache()

# %%
