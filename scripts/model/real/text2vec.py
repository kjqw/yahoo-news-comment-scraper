# %%
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))

import torch
from db_manager import execute_query
from transformers import pipeline


# %%
def classify(
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
            multi_label=True,
        )

        return results

    except Exception as e:
        pass

    finally:
        # メモリ解放
        del classifier
        torch.cuda.empty_cache()


def load_data(db_congif: dict):
    pass


# %%
# モデルと入力データの設定
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

db_config = {
    "host": "postgresql_db",
    "database": "yahoo_news",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}

# コメント数が多い順にユーザーページのリンクを取得
user_links = [
    i[0]
    for i in execute_query(
        """
        SELECT user_link
        FROM users
        ORDER BY total_comment_count DESC;
        """,
        db_config,
    )
]
# %%
user_data = defaultdict(list)
for user_link in tqdm(user_links):
    comments = execute_query(
        f"""
        SELECT comment_id, article_id, parent_comment_id, comment_content, posted_time, scraped_time
        FROM comments
        WHERE user_link = '{user_link}'
        ORDER BY scraped_time DESC;
        """,
        db_config,
    )
    for comment in comments:
        data_dict = {
            "comment_content": None,
            "parent_article_title": None,
            "parent_article_content": None,
            "parent_comment_content": None,
            "posted_time": None,
            "scraped_time": None,
        }
        data_dict["comment_content"] = comment[3]
        data_dict["post_time"] = comment[4]
        data_dict["scraped_time"] = comment[5]

        # 記事の内容とタイトルを取得
        article_id = comment[1]
        data_dict["parent_article_content"] = execute_query(
            f"SELECT article_content FROM articles WHERE article_id = '{article_id}';",
            db_config,
        )[0][0]
        data_dict["parent_article_title"] = execute_query(
            f"SELECT article_title FROM articles WHERE article_id = '{article_id}';",
            db_config,
        )[0][0]

        # 親コメントを取得（存在する場合）
        parent_comment_id = comment[2]
        if parent_comment_id:
            data_dict["parent_comment_content"] = execute_query(
                f"SELECT comment_content FROM comments WHERE comment_id = '{parent_comment_id}';",
                db_config,
            )[0][0]

        # ユーザーのデータに追加
        user_data[user_link].append(data_dict)

# %%
labels_category = [
    "国内",
    "国際",
    "経済",
    "エンタメ",
    "スポーツ",
    "IT・科学",
]
labels_sentiment = ["ポジティブ", "中立", "ネガティブ"]
hypothesis_template_category = "この文章は{}に関する内容です。"
hypothesis_template_sentiment = "この文章の感情は{}です。"

# %%
dfs = {}
for user_link, data in tqdm(user_data.items()):
    parent_article_contents, parent_comment_contents = zip(
        *[(d["parent_article_content"], d["parent_comment_content"]) for d in data]
    )
    try:
        # ゼロショット分類
        results_category = classify(
            MODEL_NAME,
            list(parent_article_contents),
            labels_category,
            hypothesis_template_category,
        )
        results_sentiment = classify(
            MODEL_NAME,
            list(parent_article_contents),
            labels_sentiment,
            hypothesis_template_sentiment,
        )

        # データ保存用のデータフレームを作成
        df = pd.DataFrame(
            columns=labels_category
            + labels_sentiment
            + [
                "コメント内容",
                "記事タイトル",
                "記事内容",
                "親コメント内容",
                "投稿時間",
                "スクレイピング時間",
            ]
        )

        for result_category, result_sentiment, d in zip(
            results_category, results_sentiment, data
        ):
            # ラベルとスコアを辞書にまとめる
            label_score_map_category = dict(
                zip(result_category["labels"], result_category["scores"])
            )
            label_score_map_sentiment = dict(
                zip(result_sentiment["labels"], result_sentiment["scores"])
            )

            # 指定された順序で並べ替え
            sorted_scores_category = [
                label_score_map_category[label] for label in labels_category
            ]
            sorted_scores_sentiment = [
                label_score_map_sentiment[label] for label in labels_sentiment
            ]

            # データフレームに追加
            new_row = pd.Series(
                sorted_scores_category
                + sorted_scores_sentiment
                + [
                    d["comment_content"],
                    d["parent_article_title"],
                    d["parent_article_content"],
                    d["parent_comment_content"],
                    d["post_time"],
                    d["scraped_time"],
                ],
                index=df.columns,
            )
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

        dfs[user_link] = df

    except:
        pass


# %%
# とりあえずpickleで保存
# TODO データベースへの保存に変更する
save_path = Path(__file__).parent / "data/dfs2.pkl"
save_path.parent.mkdir(exist_ok=True, parents=True)
with save_path.open("wb") as f:
    pickle.dump(dfs, f)
# %%
# データの読み込み
save_path = Path(__file__).parent / "data/dfs2.pkl"
with save_path.open("rb") as f:
    dfs = pickle.load(f)

# %%
df_stats = pd.DataFrame(columns=labels_category)
for user_link, df in dfs.items():
    df_stats = pd.concat([df_stats, df.iloc[:, :6].sum(axis=0).to_frame().T])

# 平均と標準偏差を行に追加。行インデックスでみやすくするために行名を変更
df_stats = pd.concat(
    [
        df_stats,
        df_stats.mean(axis=0).to_frame("mean").T,
        df_stats.std(axis=0).to_frame("var").T,
    ]
)

# %%
df_stats
# %%
dfs[user_links[-1]]
# %%
