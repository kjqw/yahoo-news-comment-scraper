# %%
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
        状態の次元数
    is_discrete : bool
        離散化するかどうか
    """

    def __init__(self, state_dim: int, is_discrete: bool):
        super(Model, self).__init__()
        # 親記事の状態用のパラメータ
        self.W_p = nn.Parameter(torch.randn(state_dim, state_dim))
        # 親コメントの状態用のパラメータ
        self.W_q = nn.Parameter(torch.randn(state_dim, state_dim))
        # 前回の状態用のパラメータ
        self.W_s = nn.Parameter(torch.randn(state_dim, state_dim))
        # バイアス
        self.b = nn.Parameter(torch.randn(state_dim, 1))
        self.is_discrete = is_discrete

    def forward(
        self,
        parent_article_state: torch.Tensor,
        parent_comment_state: torch.Tensor | None,
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        順伝播の計算を行う。

        Parameters
        ----------
        parent_article_state : torch.Tensor
            親記事の状態
        parent_comment_state : torch.Tensor | None
            親コメントの状態
        previous_state : torch.Tensor
            前回の状態

        Returns
        -------
        torch.Tensor
            予測された次の状態
        """
        # 親コメントがNoneの場合
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

        # 離散化がONの場合は閾値で区切る
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
        訓練用のデータセット
    state_dim : int
        状態の次元数
    is_discrete : bool, optional
        モデルが離散化するかどうか
    batch_size : int, optional
        バッチサイズ
    num_epochs : int, optional
        訓練エポック数
    """
    # CUDAが使えるかどうかを判定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルを初期化
    model = Model(state_dim, is_discrete).to(device)

    # 損失関数、オプティマイザを設定
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # DataLoaderの用意
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # エポックごとの損失を保存するリスト
    epoch_losses = []

    # エポックループにtqdmを使い、leave=Falseで終了時に表示を消す
    epoch_pbar = tqdm(range(num_epochs), desc="Training (Epochs)", leave=False)
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0

        # バッチ単位のループはtqdmではなく通常のforループで回す
        for batch in dataloader:
            article_states, parent_comment_states, comment_states, masks = [
                data.to(device) for data in batch
            ]

            # 前の状態は前のコメント状態として初期化
            previous_states = comment_states.clone()

            # 順伝播
            predicted_states = model(
                parent_article_state=article_states,
                parent_comment_state=parent_comment_states,
                previous_state=previous_states,
            )

            # 損失計算（マスクを適用）
            losses = criterion(predicted_states, comment_states)
            masked_losses = losses * masks.unsqueeze(-1)
            loss = masked_losses.mean()

            # 勾配リセット、逆伝播、パラメータ更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # バッチの損失を累計
            epoch_loss += loss.item()

        # エポックごとの損失を保存
        epoch_losses.append(epoch_loss)

        # 同じ行にエポックと損失を更新表示
        epoch_pbar.set_postfix({"Epoch": epoch + 1, "Loss": f"{epoch_loss:.4f}"})

    # エポックバーを終了させる
    epoch_pbar.close()

    # 学習が終わったら最初、最後、平均の損失を表示
    first_loss = epoch_losses[0] if len(epoch_losses) > 0 else float("nan")
    final_loss = epoch_losses[-1] if len(epoch_losses) > 0 else float("nan")
    mean_loss = (
        sum(epoch_losses) / len(epoch_losses) if len(epoch_losses) > 0 else float("nan")
    )
    print(
        f"First Loss: {first_loss:.4f}, Final Loss: {final_loss:.4f}, Mean Loss: {mean_loss:.4f}"
    )


def main() -> None:
    """
    メイン関数。
    ユーザーIDごとにデータを取り出してモデルを学習させる。
    """
    # 状態の次元数（記事ベクトルの次元）
    state_dim = len(df["article_content_vector"].iloc[0])
    # 離散化するかどうか
    is_discrete = True

    # ユーザーIDのユニークリストを取得
    user_ids = df["user_id"].unique()

    # ユーザーごとの処理をtqdmで進捗表示
    for user_id in tqdm(user_ids, desc="Processing Users"):
        # 該当ユーザーのデータを抽出
        df_tmp = df[df["user_id"] == user_id]

        # 投稿時間でソート（同一時刻の場合はcomment_idが新しい順）
        df_sorted = df_tmp.sort_values(
            by=["normalized_posted_time", "comment_id"],
            ascending=[True, False],
        )

        # 親コメントベクトルとマスクを作る
        parent_comment_vectors = []
        parent_comment_masks = []

        # Noneの場合はゼロベクトルを入れ、maskは0
        # Noneでない場合は実際のベクトルを入れ、maskは1
        for vec in df_sorted["parent_comment_content_vector"].values:
            if vec is None:
                parent_comment_vectors.append(
                    torch.zeros_like(
                        torch.tensor(
                            df["article_content_vector"].iloc[0], dtype=torch.float32
                        )
                    )
                )
                parent_comment_masks.append(0)
            else:
                parent_comment_vectors.append(torch.tensor(vec, dtype=torch.float32))
                parent_comment_masks.append(1)

        # TensorDatasetを用意
        dataset = TensorDataset(
            torch.tensor(
                list(df_sorted["article_content_vector"].values),
                dtype=torch.float32,
                requires_grad=True,
            ),
            torch.stack(parent_comment_vectors),  # requires_gradはFalseのままでOK
            torch.tensor(
                list(df_sorted["comment_content_vector"].values),
                dtype=torch.float32,
                requires_grad=True,
            ),
            torch.tensor(parent_comment_masks, dtype=torch.float32),
        )

        # モデルの学習を実行
        train(
            dataset=dataset,
            state_dim=state_dim,
            is_discrete=is_discrete,
            batch_size=8,
            num_epochs=10,
        )


# %%
if __name__ == "__main__":
    main()
# %%
