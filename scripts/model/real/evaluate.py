# %%
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# データベースモジュールのパスをシステムパスに追加
sys.path.append(str(Path(__file__).parents[2]))
from db_manager import execute_query

# %%
# データベース接続設定
DATABASE_CONFIG = {
    "host": "postgresql_db",
    "database": "yahoo_news_modeling_1",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}

# データベースからデータを取得
TRAINING_DATA_QUERY = """
    SELECT *
    FROM training_data_vectorized_sentiment
"""
training_data_vectorized_sentiment = execute_query(TRAINING_DATA_QUERY, DATABASE_CONFIG)

# カラム名を定義し、データをDataFrameに変換
COLUMN_NAMES = [
    "user_id",
    "article_id",
    "article_content_vector",
    "parent_comment_id",
    "parent_comment_content_vector",
    "comment_id",
    "comment_content_vector",
    "normalized_posted_time",
]
df = pd.DataFrame(training_data_vectorized_sentiment, columns=COLUMN_NAMES)

# ユーザーIDの一覧を取得
user_ids = df["user_id"].unique()

# %%
"""
`scripts/scraping/main_user.py`の
# user_linkからuser_idを特定して、commentsテーブルにuser_idを追加
のセルを実行した際に余計なuser_idが追加されているので、それを削除する。
コメント数が1のものが紛れ込んでおり、1ステップ前と後のデータを取得できないので訓練時にエラーが起きた。
"""
# 出現回数が100以上のものをフィルタリング
value_counts = df["user_id"].value_counts()
filtered_users = value_counts[value_counts >= 100]
user_ids = filtered_users.index
user_ids

# %%
value_counts
# %%
DATA_PATH = Path(__file__).parent / "data"
RESULT_PATH = DATA_PATH / "predicition_results"

# TODO 手動で指定は面倒
DATA_PATHS = [
    # * ランダムなデータで学習
    # (RESULT_PATH / "linear", DATA_PATH / "20250122170943"),  # linear
    # (RESULT_PATH / "diff", DATA_PATH / "20250122172941"),  # diff
    # (RESULT_PATH / "nn", DATA_PATH / "20250122165142"),  # nn
    # * 過去のデータで学習、最新のデータで評価
    (RESULT_PATH / "1/linear", DATA_PATH / "20250126161128"),  # linear
    (RESULT_PATH / "1/diff", DATA_PATH / "20250126162046"),  # diff
    (RESULT_PATH / "1/nn", DATA_PATH / "20250126162942"),  # nn
    # * linear, diff モデルのtanhをsoftmaxに変更
    # (RESULT_PATH / "2/linear", DATA_PATH / "20250127110129"),  # linear
    # (RESULT_PATH / "2/diff", DATA_PATH / "20250127111017"),  # diff
    # (RESULT_PATH / "2/nn", DATA_PATH / "20250126162942"),  # nn
]

result_dict = {}
for result_path, data_path in DATA_PATHS:
    for user_id in user_ids:
        # モデルを読み込む
        model = torch.load(f"{data_path}/models/model_{user_id}.pt")
        model.to("cpu")
        model.eval()

        df_user = df[df["user_id"] == user_id]
        # df_user = df_user[: int(0.8 * len(df_user))]  # 前半の訓練データを使用
        df_user = df_user[int(0.8 * len(df_user)) :]  # 後半の評価データを使用

        pred_states = [[0, 0, 0]]
        for (
            article_content_vector,
            parent_comment_content_vector,
            comment_content_vector,
            comment_content_vector_next,
        ) in zip(
            df_user["article_content_vector"][:-1],
            df_user["parent_comment_content_vector"][:-1],
            df_user["comment_content_vector"][:-1],
            df_user["comment_content_vector"][1:],
        ):
            article_content_vector = torch.tensor(
                article_content_vector, dtype=torch.float32
            )
            parent_comment_content_vector = (
                torch.tensor(parent_comment_content_vector, dtype=torch.float32)
                if parent_comment_content_vector is not None
                else torch.tensor([0, 0, 0], dtype=torch.float32)
            )
            comment_content_vector = torch.tensor(
                comment_content_vector, dtype=torch.float32
            )
            comment_content_vector_next = torch.tensor(
                comment_content_vector_next, dtype=torch.float32
            )

            pred_state = model(
                article_content_vector,
                parent_comment_content_vector,
                comment_content_vector,
            )

            if model.__class__.__name__ == "NNModel":
                pred_states.append(pred_state.tolist())
            else:
                pred_states.append(pred_state.tolist()[0])

        df_user["pred_state"] = pred_states

        # comment_content_vectorとpred_stateの列をnp.arrayに変換
        df_user["comment_content_vector"] = df_user["comment_content_vector"].apply(
            lambda x: np.array(x)
        )
        df_user["pred_state"] = df_user["pred_state"].apply(lambda x: np.array(x))

        # 実測値と予測値の差を計算
        df_user["cosine_similarity"] = df_user.apply(
            lambda x: cosine_similarity(
                x["comment_content_vector"].reshape(1, -1),
                x["pred_state"].reshape(1, -1),
            )[0][0],
            axis=1,
        )
        df_user["euclidean_distance"] = df_user.apply(
            lambda x: np.linalg.norm(x["comment_content_vector"] - x["pred_state"]),
            axis=1,
        )

        # ユーザーごとの結果を保存
        result_dict[f"{result_path}_{user_id}"] = df_user


# %%
diffs = defaultdict(list)
for key, df_user in result_dict.items():
    tmp = key.split("/")[-1]
    tmp_key = tmp.split("_")[0]
    cos_mean, cos_std, euc_mean, euc_std = (
        df_user["cosine_similarity"].mean(),
        df_user["cosine_similarity"].std(),
        df_user["euclidean_distance"].mean(),
        df_user["euclidean_distance"].std(),
    )
    diffs[tmp_key].append((cos_mean, cos_std, euc_mean, euc_std))

    print(f"cosine_similarity mean: {df_user['cosine_similarity'].mean()}")
    print(f"cosine_similarity std: {df_user['cosine_similarity'].std()}")
    print(f"euclidean_distance mean: {df_user['euclidean_distance'].mean()}")
    print(f"euclidean_distance std: {df_user['euclidean_distance'].std()}")
    print()
    break

# %%
print(f"cosine_similarity mean: {df_user['cosine_similarity'].mean()}")
print(f"cosine_similarity std: {df_user['cosine_similarity'].std()}")
print(f"euclidean_distance mean: {df_user['euclidean_distance'].mean()}")
print(f"euclidean_distance std: {df_user['euclidean_distance'].std()}")

# %%
