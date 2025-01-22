# %%
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

sys.path.append(str(Path(__file__).parents[3]))

from db_manager import execute_query

# %%
# データベースの初期化
db_config = {
    "host": "postgresql_db",
    "database": "yahoo_news_modeling_1",
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
def vec2scalar(vec: list) -> float:
    """
    ポジティブ・中立・ネガティブの確率を値に持つ、和が1の3次元ベクトルを受け取り、1次元のスカラー値に変換する。

    Parameters
    ----------
    vec : list
        ポジティブ・中立・ネガティブの確率を値に持つ、和が1の3次元ベクトル。

    Returns
    -------
    float
        スカラー値。

    Examples
    --------
    >>> vec2scalar([x, y, z])
    x - z
    >>> vec2scalar([0.3, 0.6, 0.1])
    0.2
    """
    return float(vec[0] - vec[2])


# %%
df_training_data_vectorized_sentiment["comment_sentiment_scalar"] = (
    df_training_data_vectorized_sentiment["comment_content_vector"].apply(vec2scalar)
)
# %%
value_counts = df_training_data_vectorized_sentiment["user_id"].value_counts()
filtered_users = value_counts[value_counts >= 100]
user_ids = filtered_users.index

# %%
value_counts
# %%
for user_id in user_ids:
    df_user = df_training_data_vectorized_sentiment[
        df_training_data_vectorized_sentiment["user_id"] == user_id
    ]

    fig, ax = plt.subplots()

    ax.set_title(f"user_id: {user_id}")
    ax.set_xlabel("posted time")
    ax.set_ylabel("comment sentiment scalar")
    ax.set_ylim(-1.1, 1.1)
    ax.tick_params(axis="x", rotation=45)

    ax.plot(df_user["normalized_posted_time"], df_user["comment_sentiment_scalar"])
# %%
