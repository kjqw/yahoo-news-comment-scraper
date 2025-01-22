# %%
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from models import DiffModel, LinearModel, NNModel
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

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
# %%


# %%
def split_dataset(dataset: TensorDataset, split_ratio: float):
    """
    データセットを訓練用と評価用に分割する。

    Parameters
    ----------
    dataset : TensorDataset
        分割対象のデータセット。
    split_ratio : float
        訓練データセットの割合（例: 0.8 で 80% が訓練用）。

    Returns
    -------
    tuple[Subset, Subset]
        訓練用データセットと評価用データセット。
    """
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


def train_and_evaluate(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    num_epochs: int = 1000,
) -> tuple[list[float], list[float]]:
    """
    モデルの訓練と評価を行い、損失の履歴を返す。

    Parameters
    ----------
    train_loader : DataLoader
        訓練用データローダ。
    val_loader : DataLoader
        評価用データローダ。
    model : LinearModel
        学習対象のモデル。
    num_epochs : int, optional
        学習エポック数。

    Returns
    -------
    tuple[list[float], list[float]]
        訓練損失と評価損失の履歴。
    """
    criterion = nn.MSELoss()
    optimizer = RAdamScheduleFree(model.parameters(), lr=0.001)

    train_loss_history = []
    val_loss_history = []

    model.train()  # モデルを訓練モードに設定
    optimizer.train()  # オプティマイザを訓練モードに設定
    # 学習率スケジューリングを勝手にやってくれるらしい
    # https://zenn.dev/dena/articles/6f04641801b387

    with tqdm(range(num_epochs)) as progress_bar:
        for epoch in progress_bar:
            epoch_train_loss = 0.0

            # 訓練データのミニバッチ処理
            for (
                parent_article_state,
                parent_comment_state,
                previous_state,
                next_state,
            ) in train_loader:
                optimizer.zero_grad()
                pred_state = model(
                    parent_article_state, parent_comment_state, previous_state
                )
                loss = criterion(pred_state, next_state)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            # エポックごとの訓練損失を記録
            average_train_loss = epoch_train_loss / len(train_loader)
            train_loss_history.append(average_train_loss)

            # 評価モードに切り替えて評価データの損失を計算
            model.eval()
            with torch.no_grad():
                epoch_val_loss = 0.0
                for (
                    parent_article_state,
                    parent_comment_state,
                    previous_state,
                    next_state,
                ) in val_loader:
                    pred_state = model(
                        parent_article_state, parent_comment_state, previous_state
                    )
                    loss = criterion(pred_state, next_state)
                    epoch_val_loss += loss.item()

                # エポックごとの評価損失を記録
                average_val_loss = epoch_val_loss / len(val_loader)
                val_loss_history.append(average_val_loss)

            model.train()  # 訓練モードに戻す
            progress_bar.set_postfix(
                {"train_loss": average_train_loss, "val_loss": average_val_loss}
            )

    return train_loss_history, val_loss_history


# %%
# モデルの設定と学習
STATE_DIM = 3
IS_DISCRETE = False
BATCH_SIZE = 8
NUM_EPOCHS = 500
SPLIT_RATIO = 0.8
SETTINGS = {
    "state_dim": STATE_DIM,
    "is_discrete": IS_DISCRETE,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "split_ratio": SPLIT_RATIO,
    "database_config": DATABASE_CONFIG,
}

# モデルと損失の履歴を保存するディレクトリ
TIME_NOW = datetime.now().strftime("%Y%m%d%H%M%S")
DATA_PATH = Path(__file__).parent / f"data/{TIME_NOW}"
MODEL_PATH = DATA_PATH / f"models"
LOSS_HISTORIES_PATH = DATA_PATH / "loss_histories"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ保存用のディレクトリを作成
MODEL_PATH.mkdir(parents=True, exist_ok=True)
LOSS_HISTORIES_PATH.mkdir(parents=True, exist_ok=True)

for user_id in user_ids:
    # ユーザーごとのデータを取得し、時間順に並び替え
    user_data = df[df["user_id"] == user_id]
    user_data_sorted = user_data.sort_values(
        by=["normalized_posted_time", "comment_id"], ascending=[True, False]
    )

    # データセットの作成
    dataset = TensorDataset(
        torch.tensor(
            [i for i in user_data_sorted["article_content_vector"][:-1]],
            dtype=torch.float32,
        ).to(DEVICE),
        torch.tensor(
            [
                i if i is not None else [2, 2, 2]
                for i in user_data_sorted["parent_comment_content_vector"][:-1]
            ],
            dtype=torch.float32,
        ).to(DEVICE),
        torch.tensor(
            [i for i in user_data_sorted["comment_content_vector"][:-1]],
            dtype=torch.float32,
        ).to(DEVICE),
        torch.tensor(
            [i for i in user_data_sorted["comment_content_vector"][1:]],
            dtype=torch.float32,
        ).to(DEVICE),
    )

    # データセットを訓練用と評価用に分割
    train_dataset, val_dataset = split_dataset(dataset, SPLIT_RATIO)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # モデルを初期化
    # model = LinearModel(STATE_DIM, IS_DISCRETE).to(DEVICE)
    # model = DiffModel(STATE_DIM, IS_DISCRETE).to(DEVICE)
    model = NNModel(STATE_DIM, IS_DISCRETE, [128, 128]).to(DEVICE)

    # モデルを訓練し、損失の履歴を取得
    train_loss, val_loss = train_and_evaluate(
        train_loader, val_loader, model, num_epochs=NUM_EPOCHS
    )

    # 損失の履歴を保存
    with open(LOSS_HISTORIES_PATH / f"loss_histories_{user_id}.json", "w") as f:
        json.dump({"train_loss": train_loss, "val_loss": val_loss}, f, indent=4)
    # モデルを保存
    torch.save(model, MODEL_PATH / f"model_{user_id}.pt")

SETTINGS["model"] = model.__class__.__name__
# 設定を保存
with open(DATA_PATH / "settings.json", "w") as f:
    json.dump(SETTINGS, f, indent=4)

# %%
