import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
class Model(nn.Module):
    """
    状態モデルクラス。
    親記事、親コメント、前の状態から次の状態を予測する。

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
                pred_state > 0.5, 1, torch.where(pred_state < -0.5, -1, 0)
            )

        return pred_state


def train(
    dataset: Dataset,
    state_dim: int,
    is_discrete: bool = True,
    batch_size: int = 8,
    num_epochs: int = 100,
) -> None:
    """
    モデルの訓練を行う関数。

    Parameters
    ----------
    dataset : Dataset
        訓練用のデータセット。
    state_dim : int
        状態の次元数。
    is_discrete : bool, optional
        モデルが離散化するかどうか。
    batch_size : int, optional
        バッチサイズ。
    num_epochs : int, optional
        訓練エポック数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(state_dim, is_discrete).to(device)
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        epoch_loss = 0.0

        with tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ) as pbar:
            for batch in pbar:
                article_states, parent_comment_states, comment_states, masks = [
                    data.to(device) for data in batch
                ]

                # 前の状態は前のコメント状態として初期化
                previous_states = comment_states.clone()

                # モデルの順伝播
                predicted_states = model(
                    parent_article_state=article_states,
                    parent_comment_state=parent_comment_states,
                    previous_state=previous_states,
                )

                # マスクを利用して損失を計算
                losses = criterion(predicted_states, comment_states)
                masked_losses = losses * masks.unsqueeze(-1)
                loss = masked_losses.mean()

                # パラメータの更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 現在のバッチの損失を表示
                pbar.set_postfix({"Batch Loss": loss.item()})
                epoch_loss += loss.item()

        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# %%
# ユーザーごとのデータでモデルを学習する例
state_dim = len(df["article_content_vector"].iloc[0])  # 状態の次元数を設定
is_discrete = True  # モデルが離散化する場合はTrue

# ユーザーIDのユニークリストを取得
user_ids = df["user_id"].unique()

# ユーザーごとに処理
for user_id in tqdm(user_ids, desc="Processing Users"):
    # 該当ユーザーのデータを抽出
    df_tmp = df[df["user_id"] == user_id]

    # 投稿時間でソート（同一時刻の場合はcomment_idが新しい順）
    df_sorted = df_tmp.sort_values(
        by=["normalized_posted_time", "comment_id"],
        ascending=[True, False],
    )

    # 親コメントベクトルとマスクを作成
    parent_comment_vectors = []
    parent_comment_masks = []

    for vec in df_sorted["parent_comment_content_vector"].values:
        if vec is None:
            parent_comment_vectors.append(
                torch.zeros_like(
                    torch.tensor(
                        df["article_content_vector"].iloc[0], dtype=torch.float32
                    )
                )
            )
            parent_comment_masks.append(0)  # maskは0
        else:
            parent_comment_vectors.append(
                torch.tensor(vec, dtype=torch.float32, requires_grad=True)
            )
            parent_comment_masks.append(1)  # maskは1

    # データセットを作成
    dataset = TensorDataset(
        torch.tensor(
            list(df_sorted["article_content_vector"].values),
            dtype=torch.float32,
            requires_grad=True,
        ),
        torch.stack(parent_comment_vectors),
        torch.tensor(
            list(df_sorted["comment_content_vector"].values),
            dtype=torch.float32,
            requires_grad=True,
        ),
        torch.tensor(parent_comment_masks, dtype=torch.float32),
    )

    # 学習の実行
    # print(f"Training model for user_id: {user_id}")
    tqdm.write(f"Training model for user_id: {user_id}")
    train(
        dataset=dataset,
        state_dim=state_dim,
        is_discrete=is_discrete,
        batch_size=8,
        num_epochs=10,
    )

# %%
