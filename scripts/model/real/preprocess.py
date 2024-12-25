# %%
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from db_manager import execute_query

# from llm import zeroshot

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
# スクレイピング結果をLLMでベクトル化
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
user_ids, _ = zip(
    *execute_query(
        """
        SELECT DISTINCT users.user_id, users.total_comment_count
        FROM users
        INNER JOIN comments ON users.user_id = comments.user_id
        ORDER BY users.total_comment_count DESC;
        """,
        db_config,
    )
)
user_ids
# %%
# TODO: forでクエリを繰り返しているため遅い
user_data = defaultdict(dict)
for user_id in user_ids:
    comment_ids, article_ids, parent_comment_ids, normalized_posted_times = zip(
        *execute_query(
            f"""
            SELECT comment_id, article_id, parent_comment_id, normalized_posted_time
            FROM comments
            WHERE user_id = {user_id}
            ORDER BY normalized_posted_time ASC, scraped_time ASC;
            """,
            db_config,
        )
    )
    for i, (
        comment_id,
        article_id,
        parent_comment_id,
        normalized_posted_time,
    ) in enumerate(
        zip(comment_ids, article_ids, parent_comment_ids, normalized_posted_times)
    ):
        comment_content = execute_query(
            f"""
            SELECT comment_content
            FROM comments
            WHERE comment_id = {comment_id};
            """,
            db_config,
        )[0][0]
        article_content = execute_query(
            f"""
            SELECT article_content
            FROM articles
            WHERE article_id = {article_id};
            """,
            db_config,
        )[0][0]
        if parent_comment_id is None:
            user_data[user_id][i] = {
                "comment_content": comment_content,
                "article_content": article_content,
                "parent_comment_content": None,
                "normalized_posted_time": normalized_posted_time,
            }
        else:
            parent_comment_content = execute_query(
                f"""
                SELECT comment_content
                FROM comments
                WHERE comment_id = {parent_comment_id};
                """,
                db_config,
            )[0][0]

            user_data[user_id][i] = {
                "comment_content": comment_content,
                "article_content": article_content,
                "parent_comment_content": parent_comment_content,
                "normalized_posted_time": normalized_posted_time,
            }

# %%
user_id
# %%
user_data

# %%
user_data.keys()
# %%
len(user_data[8])
# %%
# 結果をデータベースに保存
# %%
