# %%
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))

from db_manager import execute_query

# %%
# データベースの初期化
db_config = {
    "host": "postgresql_db",
    "database": "yahoo_news_restore",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}
# %%
training_data_vectorized_sentiment = execute_query(
    """
    SELECT *
    FROM training_data_vectorized_sentiment
    """,
    db_config,
)

# %%
column_names = [
    "user_id",
    "article_id",
    "article_content_vector",
    "parent_comment_id",
    "parent_comment_content_vector",
    "comment_id",
    "comment_content_vector",
    "normalized_posted_time",
]
# %%
df_training_data_vectorized_sentiment = pd.DataFrame(
    training_data_vectorized_sentiment, columns=column_names
)
# %%
df_training_data_vectorized_sentiment
# %%
print(
    df_training_data_vectorized_sentiment[
        [
            "article_content_vector",
            "parent_comment_content_vector",
            "comment_content_vector",
        ]
    ].head()
)
# %%
