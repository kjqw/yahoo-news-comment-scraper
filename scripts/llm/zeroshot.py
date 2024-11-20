import torch
from transformers import pipeline


def main(
    model_name: str, texts: list[str], labels: list[str], hypothesis_template: str
):
    try:
        # 使用するデバイスを設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # モデルをロード
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if device.type == "cuda" else -1,
        )

        # ゼロショット分類
        results = classifier(
            texts,
            labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )

        return results

    except Exception as e:
        print(e)

    finally:
        # メモリ解放
        del classifier
        torch.cuda.empty_cache()


if __name__ == "__main__":

    # モデルと入力データの設定
    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    texts = [
        "新しいiPhoneの価格が高すぎると批判されています。",
        "日本代表がワールドカップで優勝しました！",
        "映画のレビューが酷評されていました。",
        "新しいスマートフォンが発売されました。",
    ]
    labels_category = ["スマートフォン", "エンタメ", "スポーツ"]
    labels_sentiment = ["ポジティブ", "中立", "ネガティブ"]
    hypothesis_template_category = "この文章は{}に関する内容です。"
    hypothesis_template_sentiment = "この文章の感情は{}です。"

    # ゼロショット分類

    results_category = main(
        MODEL_NAME,
        texts,
        labels_category,
        hypothesis_template_category,
    )
    results_sentiment = main(
        MODEL_NAME,
        texts,
        labels_sentiment,
        hypothesis_template_sentiment,
    )

    # 結果の表示
    print("Category:")
    for i, result in enumerate(results_category):
        print(f"Text: {texts[i]}")
        print(f"Category: {result['labels'][0]}")
        print(f"Score: {result['scores'][0]:.2f}")
        print()

    print("Sentiment:")
    for i, result in enumerate(results_sentiment):
        print(f"Text: {texts[i]}")
        print(f"Sentiment: {result['labels'][0]}")
        print(f"Score: {result['scores'][0]:.2f}")
        print()
