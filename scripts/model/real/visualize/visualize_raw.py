# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))

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
user_ids = df_training_data_vectorized_sentiment["user_id"].unique()
user_ids
# %%
df_posnegs = {}
for user_id in user_ids:
    df_user = df_training_data_vectorized_sentiment[
        (df_training_data_vectorized_sentiment["user_id"] == user_id)
    ]
    article_content_vectors, parent_comment_content_vectors, comment_content_vectors = (
        df_user["article_content_vector"].values.tolist(),
        df_user["parent_comment_content_vector"].values.tolist(),
        df_user["comment_content_vector"].values.tolist(),
    )
    article_content_posnegs, parent_comment_content_posnegs, comment_content_posnegs = (
        [row.index(max(row)) for row in article_content_vectors],
        [
            row.index(max(row)) if row else None
            for row in parent_comment_content_vectors
        ],
        [row.index(max(row)) for row in comment_content_vectors],
    )
    df_posnegs[user_id] = pd.DataFrame(
        {
            "article_content_posneg": article_content_posnegs,
            "parent_comment_content_posneg": parent_comment_content_posnegs,
            "comment_content_posneg": comment_content_posnegs,
        }
    )
    # ポジティブを1、中立を0、ネガティブを-1に変換
    mapping = {2: -1, 1: 0, 0: 1}

    df_posnegs[user_id] = df_posnegs[user_id].applymap(lambda x: mapping.get(x, x))

# %%
df_posnegs.keys()
# %%
df_posnegs[41]
# %%
df_posnegs[41][
    # df_posnegs[41]["parent_comment_content_posneg"].isnull()
    df_posnegs[41]["article_content_posneg"].isnull()
]
# %%
sns.scatterplot(
    data=df_posnegs[41],
    x=df_posnegs[41].index,
    y="comment_content_posneg",
    hue="article_content_posneg",
)
# %%
