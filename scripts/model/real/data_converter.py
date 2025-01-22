"""
training_data_raw テーブルの文章をLLMで数値化して training_data_vectorized テーブルに保存する処理
"""

# %%
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))

from db_manager import execute_query
from llm import zeroshot

# %%
# データベースの初期化
db_config = {
    "host": "postgresql_db",
    "database": "yahoo_news_modeling_1",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}

init_sql_path = Path(__file__).parent / "init.sql"
with init_sql_path.open() as f:
    query = f.read()
execute_query(query, db_config=db_config, commit=True)

# %%
# 数値化するためのLLMの設定
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
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
user_ids = [
    i[0]
    for i in execute_query(
        """
        SELECT DISTINCT user_id
        FROM training_data_raw
        """,
        db_config,
    )
]

# %%
raw_data = {}
for user_id in user_ids:
    raw_data[user_id] = execute_query(
        f"""
        SELECT *
        FROM training_data_raw
        WHERE user_id = {user_id}
        ORDER BY normalized_posted_time ASC;
        """,
        db_config,
    )

# %%
# ベクトル化してデータベースに保存
order_sentiment = ["ポジティブ", "中立", "ネガティブ"]
order_category = ["国内", "国際", "経済", "エンタメ", "スポーツ", "IT・科学"]
for user_id, data in tqdm(raw_data.items()):
    (
        _,
        article_ids,
        article_contents,
        parent_comment_ids,
        parent_comment_contents,
        comment_ids,
        comment_contents,
        normalized_posted_times,
    ) = map(list, zip(*data))

    # ネガポジをベクトル化
    article_content_vector_sentiment = zeroshot.main(
        MODEL_NAME,
        article_contents,
        labels_sentiment,
        hypothesis_template_sentiment,
    )
    parent_comment_content_vector_sentiment = zeroshot.main(
        MODEL_NAME,
        parent_comment_contents,
        labels_sentiment,
        hypothesis_template_sentiment,
    )
    comment_content_vector_sentiment = zeroshot.main(
        MODEL_NAME,
        comment_contents,
        labels_sentiment,
        hypothesis_template_sentiment,
    )
    # 出力を整形
    sorted_article_content_vector_sentiment = {
        "sequences": [item["sequence"] for item in article_content_vector_sentiment],
        "labels": order_sentiment,
        "scores": [
            [item["scores"][item["labels"].index(label)] for label in order_sentiment]
            for item in article_content_vector_sentiment
        ],
    }
    sorted_parent_comment_content_vector_sentiment = {
        "sequences": [
            item["sequence"] if item is not None else None
            for item in parent_comment_content_vector_sentiment
        ],
        "labels": order_sentiment,
        "scores": [
            (
                [
                    item["scores"][item["labels"].index(label)]
                    for label in order_sentiment
                ]
                if item is not None
                else None
            )
            for item in parent_comment_content_vector_sentiment
        ],
    }
    sorted_comment_content_vector_sentiment = {
        "sequences": [item["sequence"] for item in comment_content_vector_sentiment],
        "labels": order_sentiment,
        "scores": [
            [item["scores"][item["labels"].index(label)] for label in order_sentiment]
            for item in comment_content_vector_sentiment
        ],
    }
    # 結果をデータベースに保存
    data_to_insert_sentiment = [
        (
            user_id,
            article_id,
            article_content_vector,
            parent_comment_id,
            parent_comment_content_vector,
            comment_id,
            comment_content_vector,
            normalized_posted_time,
        )
        for (
            article_id,
            article_content_vector,
            parent_comment_id,
            parent_comment_content_vector,
            comment_id,
            comment_content_vector,
            normalized_posted_time,
        ) in zip(
            article_ids,
            sorted_article_content_vector_sentiment["scores"],
            parent_comment_ids,
            sorted_parent_comment_content_vector_sentiment["scores"],
            comment_ids,
            sorted_comment_content_vector_sentiment["scores"],
            normalized_posted_times,
        )
    ]
    execute_query(
        f"""
        INSERT INTO training_data_vectorized_sentiment
        (user_id, article_id, article_content_vector, parent_comment_id, parent_comment_content_vector, comment_id, comment_content_vector, normalized_posted_time)
        VALUES
        """
        + ",".join(
            [
                (
                    """('{}', '{}', ARRAY{}, '{}', ARRAY{}, '{}', ARRAY{}, '{}')""".format(
                        d[0],
                        d[1],
                        d[2],
                        d[3],
                        d[4],
                        d[5],
                        d[6],
                        d[7],
                    )
                    if d[3] is not None
                    else """('{}', '{}', ARRAY{}, NULL, NULL, '{}', ARRAY{}, '{}')""".format(
                        d[0], d[1], d[2], d[5], d[6], d[7]
                    )
                )
                for d in data_to_insert_sentiment
            ]
        ),
        db_config,
        commit=True,
    )

    # カテゴリをベクトル化
    article_content_vector_category = zeroshot.main(
        MODEL_NAME,
        article_contents,
        labels_category,
        hypothesis_template_category,
    )
    parent_comment_content_vector_category = zeroshot.main(
        MODEL_NAME,
        parent_comment_contents,
        labels_category,
        hypothesis_template_category,
    )
    comment_content_vector_category = zeroshot.main(
        MODEL_NAME,
        comment_contents,
        labels_category,
        hypothesis_template_category,
    )
    # 出力を整形
    sorted_article_content_vector_category = {
        "sequences": [item["sequence"] for item in article_content_vector_category],
        "labels": order_category,
        "scores": [
            [item["scores"][item["labels"].index(label)] for label in order_category]
            for item in article_content_vector_category
        ],
    }
    sorted_parent_comment_content_vector_category = {
        "sequences": [
            item["sequence"] if item is not None else None
            for item in parent_comment_content_vector_category
        ],
        "labels": order_category,
        "scores": [
            (
                [
                    item["scores"][item["labels"].index(label)]
                    for label in order_category
                ]
                if item is not None
                else None
            )
            for item in parent_comment_content_vector_category
        ],
    }
    sorted_comment_content_vector_category = {
        "sequences": [item["sequence"] for item in comment_content_vector_category],
        "labels": order_category,
        "scores": [
            [item["scores"][item["labels"].index(label)] for label in order_category]
            for item in comment_content_vector_category
        ],
    }
    # 結果をデータベースに保存
    data_to_insert_category = [
        (
            user_id,
            article_id,
            article_content_vector,
            parent_comment_id,
            parent_comment_content_vector,
            comment_id,
            comment_content_vector,
            normalized_posted_time,
        )
        for (
            article_id,
            article_content_vector,
            parent_comment_id,
            parent_comment_content_vector,
            comment_id,
            comment_content_vector,
            normalized_posted_time,
        ) in zip(
            article_ids,
            sorted_article_content_vector_category["scores"],
            parent_comment_ids,
            sorted_parent_comment_content_vector_category["scores"],
            comment_ids,
            sorted_comment_content_vector_category["scores"],
            normalized_posted_times,
        )
    ]
    execute_query(
        f"""
        INSERT INTO training_data_vectorized_category
        (user_id, article_id, article_content_vector, parent_comment_id, parent_comment_content_vector, comment_id, comment_content_vector, normalized_posted_time)
        VALUES
        """
        + ",".join(
            [
                (
                    """('{}', '{}', ARRAY{}, '{}', ARRAY{}, '{}', ARRAY{}, '{}')""".format(
                        d[0],
                        d[1],
                        d[2],
                        d[3],
                        d[4],
                        d[5],
                        d[6],
                        d[7],
                    )
                    if d[3] is not None
                    else """('{}', '{}', ARRAY{}, NULL, NULL, '{}', ARRAY{}, '{}')""".format(
                        d[0], d[1], d[2], d[5], d[6], d[7]
                    )
                )
                for d in data_to_insert_category
            ]
        ),
        db_config,
        commit=True,
    )


# %%
