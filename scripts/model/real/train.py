# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader, TensorDataset
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
user_ids = df["user_id"].unique()


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
        # self.W_p = nn.Parameter(torch.randn(state_dim, state_dim)).to(device)
        # self.W_q = nn.Parameter(torch.randn(state_dim, state_dim)).to(device)
        # self.W_s = nn.Parameter(torch.randn(state_dim, state_dim)).to(device)
        # self.b = nn.Parameter(torch.randn(state_dim, 1)).to(device)
        self.W_p = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_q = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_s = nn.Parameter(torch.randn(state_dim, state_dim))
        self.b = nn.Parameter(torch.randn(state_dim, 1))
        self.is_discrete = is_discrete

    def forward(
        self,
        parent_article_state: torch.Tensor,
        parent_comment_state: torch.Tensor,
        previous_state: torch.Tensor,
    ):
        """
        順伝播の計算を行う。

        Parameters
        ----------
        parent_article_state : torch.Tensor
            親記事の状態。
        parent_comment_state : torch.Tensor
            親コメントの状態。Noneだとtorchで扱いにくいので、Noneのときは[2, 2, 2]を入力することにした。
        previous_state : torch.Tensor
            前回の状態。

        Returns
        -------
        torch.Tensor
            予測された次の状態。
        """

        if torch.all(
            parent_comment_state
            == torch.tensor([2, 2, 2], device=parent_comment_state.device)
        ):
            pred_state = torch.tanh(
                torch.matmul(parent_article_state, self.W_p.T)
                + torch.matmul(previous_state, self.W_s.T)
                + self.b.T
            )
        else:
            pred_state = torch.tanh(
                torch.matmul(parent_article_state, self.W_p.T)
                + torch.matmul(parent_comment_state, self.W_q.T)
                + torch.matmul(previous_state, self.W_s.T)
                + self.b.T
            )

        if self.is_discrete:
            pred_state = torch.where(
                pred_state > 0.5,
                torch.tensor(1.0, dtype=torch.float32, device=pred_state.device),
                torch.where(
                    pred_state < -0.5,
                    torch.tensor(-1.0, dtype=torch.float32, device=pred_state.device),
                    torch.tensor(0.0, dtype=torch.float32, device=pred_state.device),
                ),
            )

        return pred_state


def train(
    dataset: TensorDataset,
    model: Model,
    batch_size: int = 8,
    num_epochs: int = 1000,
) -> list[float]:
    """
    モデルを学習し、損失履歴を返す。

    Parameters
    ----------
    dataset : TensorDataset
        学習用データセット。
    model : Model
        学習対象のモデル。
    batch_size : int, optional
        バッチサイズ。
    num_epochs : int, optional
        エポック数。

    Returns
    -------
    list[float]
        各エポックの損失値のリスト。
    """
    criterion = nn.MSELoss()
    optimizer = RAdamScheduleFree(model.parameters(), lr=0.001)

    model.train()
    optimizer.train()

    loss_history = []

    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            epoch_loss = 0.0  # エポック内の累積損失

            for (
                parent_article_state,
                parent_comment_state,
                previous_state,
                next_state,
            ) in dataloader:
                optimizer.zero_grad()

                pred_state = model(
                    parent_article_state, parent_comment_state, previous_state
                )
                loss = criterion(pred_state, next_state)
                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            # 平均損失を計算して履歴に記録
            average_loss = epoch_loss / len(dataloader)
            loss_history.append(average_loss)

            pbar.set_postfix({"loss": average_loss})

    return loss_history


# %%
state_dim = 3
is_discrete = False
batch_size = 8
num_epochs = 500

model_path = Path(__file__).parent / "data"
loss_histories_path = model_path / "loss_histories.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_histories = {}
for user_id in user_ids:
    df_tmp = df[df["user_id"] == user_id]
    df_sorted = df_tmp.sort_values(
        by=["normalized_posted_time", "comment_id"],
        ascending=[True, False],
    )  # 投稿時間が同じ場合はcomment_idが新しい順にソート

    dataset = TensorDataset(
        torch.tensor(
            [i for i in df_sorted["article_content_vector"][:-1]],
            dtype=torch.float32,
            requires_grad=True,
        ).to(device),
        torch.tensor(
            [
                i if i is not None else [2, 2, 2]
                for i in df_sorted["parent_comment_content_vector"][:-1]
            ],
            dtype=torch.float32,
            requires_grad=True,
        ).to(device),
        torch.tensor(
            [i for i in df_sorted["comment_content_vector"][:-1]],
            dtype=torch.float32,
            requires_grad=True,
        ).to(device),
        torch.tensor(
            [i for i in df_sorted["comment_content_vector"][1:]],
            dtype=torch.float32,
            requires_grad=True,
        ).to(device),
    )

    model = Model(state_dim, is_discrete).to(device)

    # 学習
    loss_histories[user_id] = train(
        dataset,
        model,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # モデルの保存
    torch.save(model.state_dict(), model_path / f"model_{user_id}.pt")

# %%
# loss_historiesのキーを文字列に変換
loss_histories_str_keys = {str(key): value for key, value in loss_histories.items()}

# ファイルに保存
with open(loss_histories_path, "w") as f:
    json.dump(loss_histories_str_keys, f)

# %%
