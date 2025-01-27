# %%
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt

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
IMAGE_PATH = Path(__file__).parent / "images"
DATA_PATH = Path(__file__).parent / "data"

# TODO 手動で指定は面倒
DATA_PATHS = [
    # # ランダムなデータで学習
    # (IMAGE_PATH / "linear", DATA_PATH / "20250122170943"),  # linear
    # (IMAGE_PATH / "diff", DATA_PATH / "20250122172941"),  # diff
    # (IMAGE_PATH / "nn", DATA_PATH / "20250122165142"),  # nn
    # 過去のデータで学習、最新のデータで評価
    (IMAGE_PATH / "1/linear", DATA_PATH / "20250126161128"),  # linear
    (IMAGE_PATH / "1/diff", DATA_PATH / "20250126162046"),  # diff
    (IMAGE_PATH / "1/nn", DATA_PATH / "20250126162942"),  # nn
    # # linear, diff モデルのtanhをsoftmaxに変更
    # (IMAGE_PATH / "2/linear", DATA_PATH / ""),  # linear
    # (IMAGE_PATH / "2/diff", DATA_PATH / ""),  # diff
    # (IMAGE_PATH / "2/nn", DATA_PATH / ""),  # nn
]

fig_ax_dict = {}
for image_path, data_path in DATA_PATHS:
    for user_id in user_ids:
        # モデルを読み込む
        model = torch.load(f"{data_path}/models/model_{user_id}.pt")
        model.to("cpu")
        model.eval()

        df_user = df[df["user_id"] == user_id]
        # df_user = df_user[: int(0.2 * len(df_user))]  # 前半の訓練データを使用
        # df_user = df_user[int(0.8 * len(df_user)) :]  # 後半の評価データを使用

        pred_states = [[2, 2, 2]]
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

        df_user["comment_sentiment_scalar"] = df_user["comment_content_vector"].apply(
            lambda vec: float(vec[0] - vec[2])
            # lambda vec: float(vec[1])
        )
        df_user["pred_state_scalar"] = df_user["pred_state"].apply(
            lambda vec: float(vec[0] - vec[2])
            # lambda vec: float(vec[1])
        )

        fig, ax = plt.subplots()

        ax.set_title(f"user_id: {user_id}")
        ax.set_xlabel("posted time")
        ax.set_ylabel("comment sentiment scalar")
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(axis="x", rotation=45)

        # 実測値のプロット
        ax.plot(
            df_user["normalized_posted_time"],
            df_user["comment_sentiment_scalar"],
            label="actual",
        )
        # 予測値のプロット
        for i in range(1, len(df_user)):
            ax.plot(
                [
                    df_user["normalized_posted_time"].iloc[i - 1],
                    df_user["normalized_posted_time"].iloc[i],
                ],
                [
                    df_user["comment_sentiment_scalar"].iloc[i - 1],
                    df_user["pred_state_scalar"].iloc[i],
                ],
                color="red",
                linestyle="--",
                label="predicted",
            )
        # ax.plot(
        #     df_user["normalized_posted_time"][1:],
        #     df_user["pred_state_scalar"][1:],
        #     label="predicted",
        # )

        ax.legend()
        fig.tight_layout()

        fig_ax_dict[user_id] = (fig, ax)
    #     plt.close(fig)

    # # プロットを保存
    # for user_id, (fig, ax) in fig_ax_dict.items():
    #     save_path = image_path / f"{model.__class__.__name__}_{user_id}.png"
    #     save_path.parent.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(save_path)
# %%
