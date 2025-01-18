# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))
from db_manager import execute_query

# %%
# データベース設定
db_config = {
    "host": "postgresql_db",
    "database": "yahoo_news_restore",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}

# データ読み込み
training_data_vectorized_sentiment = execute_query(
    """
    SELECT *
    FROM training_data_vectorized_sentiment
    """,
    db_config,
)

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
df = pd.DataFrame(training_data_vectorized_sentiment, columns=column_names)

# %%
df.head()
# %%
user_ids = df["user_id"].unique()
# %%
# for user_id in user_ids:
#     df_tmp = df[df["user_id"] == user_id]
#     df_sorted = df_tmp.sort_values(
#         by=["normalized_posted_time", "comment_id"],
#         ascending=[True, False],
#     )  # 投稿時間が同じ場合はcomment_idが新しい順にソート
# %%


# %%
class Model(nn.Module):
    """
    状態モデルクラス。
    親記事、親コメント、前回の状態から次の状態を予測する。

    Parameters
    ----------
    state_dim : int
        状態の次元数。
    is_discrete : bool
        離散化するかどうか。
    """

    def __init__(self, state_dim, is_discrete):
        super(Model, self).__init__()
        self.W_p = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_q = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_s = nn.Parameter(torch.randn(state_dim, state_dim))
        self.b = nn.Parameter(torch.randn(state_dim, 1))
        self.is_discrete = is_discrete

    def forward(
        self,
        parent_article_state: torch.Tensor,
        parent_comment_state: torch.Tensor | None,
        previous_state: torch.Tensor,
    ):
        """
        順伝播の計算を行う。

        Parameters
        ----------
        parent_article_state : torch.Tensor
            親記事の状態。
        parent_comment_state : torch.Tensor | None
            親コメントの状態。
        previous_state : torch.Tensor
            前回の状態。

        Returns
        -------
        torch.Tensor
            予測された次の状態。
        """

        if parent_comment_state is None:
            pred_state = torch.tanh(
                self.W_p @ parent_article_state + self.W_s @ previous_state + self.b
            )
        else:
            pred_state = torch.tanh(
                self.W_p @ parent_article_state
                + self.W_q @ parent_comment_state
                + self.W_s @ previous_state
                + self.b
            )
        if self.is_discrete:
            pred_state = torch.where(
                pred_state > 0.5, 1, torch.where(pred_state < -0.5, -1, 0)
            )

        return pred_state


def train(
    data: pd.DataFrame,
    state_dim: int,
    is_discrete: bool = True,
    batch_size: int = 8,
    num_epochs: int = 100,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(state_dim, is_discrete).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


# %%
