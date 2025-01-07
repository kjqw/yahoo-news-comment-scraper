import torch
from transformers import pipeline


def main(
    model_name: str,
    texts: list[str | None],
    labels: list[str],
    hypothesis_template: str,
) -> list[dict | None]:
    """
    指定したテキストリストに対してゼロショット分類を行い、Noneを含む場合はそのままNoneを返す。

    Parameters
    ----------
    model_name : str
        使用するモデル名
    texts : list[str | None]
        分類対象のテキストリスト
    labels : list[str]
        ラベルのリスト
    hypothesis_template : str
        仮説テンプレート

    Returns
    -------
    list[dict | None]
        分類結果のリスト。Noneはそのまま保持される
    """
    try:
        # 使用するデバイスを設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # モデルをロード
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if device.type == "cuda" else -1,
        )

        # None以外のインデックスとテキストを抽出
        valid_indices_and_texts = [
            (i, text) for i, text in enumerate(texts) if text is not None
        ]
        valid_indices, valid_texts = (
            zip(*valid_indices_and_texts) if valid_indices_and_texts else ([], [])
        )

        # 有効なテキストを一括処理
        results = []
        if valid_texts:
            processed_results = classifier(
                list(valid_texts),
                labels,
                hypothesis_template=hypothesis_template,
                multi_label=False,
            )
            results = list(processed_results)

        # 元の順序で結果を構築
        final_results = [None] * len(texts)
        for idx, result in zip(valid_indices, results):
            final_results[idx] = result

        return final_results

    except Exception as e:
        print(e)
        return [None] * len(texts)

    finally:
        # メモリ解放
        if "classifier" in locals():
            del classifier
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # モデルと入力データの設定
    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    texts = [
        "【訃報】俳優の火野正平さんが死去…腰部骨折を機に体調崩す　75歳",
        "ロシア領に長距離ミサイル攻撃　米政権の容認後初　後方の弾薬庫標的か・ウクライナ",
        "「年収700万円」の地方公務員。結婚のため賃貸マンションへ申し込んだら「審査否決」と連絡が！ 借金やローンもないのになぜ？ 審査に通らない理由と対応策を解説",
        "千原せいじ　斎藤知事への発言に批判殺到　「俺は謝らへん」も反省「俺たちバカは、薄っぺらい正義感で…」",
        "「とにかく狭かった」久保建英も驚き 中国代表戦のピッチ横幅を数m狭める“奇策”に中国サッカー協会は「広さは把握していない」",
        "「え？ガチ？」グーグルマップが指示してきた驚愕のルートが話題に⇒公式もリプライ",
        "産後70kgに。ダイエットを決意した2年後の姿に…「別人」「お見事です」「ビックリしました」の声",
        "斎藤・兵庫県知事ら追及する百条委委員の兵庫県議、議員辞職　 SNSでの“中傷”原因か？",
    ]
    labels_category = [
        "国内",
        "国際",
        "経済",
        "エンタメ",
        "スポーツ",
        "IT・科学",
        "ライフ",
        "地域",
    ]
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
    for result_category, result_sentiment in zip(results_category, results_sentiment):
        print(f"Text: {result_category['sequence']}")
        print(f"Category: {result_category['labels']}")
        print(f"Score: {result_category['scores']}")
        print(f"Sentiment: {result_sentiment['labels']}")
        print(f"Score: {result_sentiment['scores']}")
        print()
