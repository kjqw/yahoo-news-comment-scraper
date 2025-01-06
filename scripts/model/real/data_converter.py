"""
training_data_raw テーブルの文章をLLMで数値化して training_data_vectorized テーブルに保存する処理
"""

# %%
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from db_manager import execute_query
from llm import zeroshot

# %%
# データベースの初期化
db_config = {
    "host": "postgresql_db",
    "database": "yahoo_news_restore",
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
        SELECT article_content, parent_comment_content, comment_content, normalized_posted_time
        FROM training_data_raw
        WHERE user_id = {user_id}
        ORDER BY normalized_posted_time ASC;
        """,
        db_config,
    )

# %%
# ベクトル化
vectorized_data_category = defaultdict(dict)
vectorized_data_sentiment = defaultdict(dict)
for user_id, data in raw_data.items():
    article_contents, parent_comment_contents, comment_contents, _ = map(
        list, zip(*data)
    )
    vectorized_data_sentiment[user_id]["article"] = zeroshot.main(
        MODEL_NAME,
        article_contents,
        labels_sentiment,
        hypothesis_template_sentiment,
    )
    vectorized_data_sentiment[user_id]["parent_comment"] = zeroshot.main(
        MODEL_NAME,
        parent_comment_contents,
        labels_sentiment,
        hypothesis_template_sentiment,
    )
    vectorized_data_sentiment[user_id]["comment"] = zeroshot.main(
        MODEL_NAME,
        comment_contents,
        labels_sentiment,
        hypothesis_template_sentiment,
    )
    break

# %%

# %%
