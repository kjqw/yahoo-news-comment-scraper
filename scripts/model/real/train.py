# %%
import json
import sys
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

# データベースモジュールのパスをシステムパスに追加
# あなたのプロジェクトで適切なパスを指定してください
sys.path.append(str(Path(__file__).parents[2]))
from db_manager import execute_query  # 新規実装はせず、既存のものを利用


###############################################################################
# データセットの作成やモデル定義など
###############################################################################
def split_dataset(
    dataset: TensorDataset,
    split_ratio: float,
    should_shuffle: bool = True,
    random_seed: int | None = None,
) -> tuple[Subset, Subset]:
    """

    データセットを訓練用と評価用に分割する。

    Parameters
    ----------
    dataset : TensorDataset
        分割対象のデータセット
    split_ratio : float
        訓練データの割合
    should_shuffle : bool, optional
        分割前にシャッフルするかどうか
    random_seed : int | None, optional
        シャッフルの乱数シード

    Returns
    -------
    tuple[Subset, Subset]
        (train_dataset, val_dataset)
    """
    dataset_size = len(dataset)
    train_size = int(dataset_size * split_ratio)

    if should_shuffle:
        if random_seed is not None:
            torch.manual_seed(random_seed)
        indices = torch.randperm(dataset_size)
    else:
        indices = torch.arange(dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


class NNModel(nn.Module):
    """

    親記事、親コメント、前回の状態を入力し、次の状態を予測するNN。

    Parameters
    ----------
    state_dim : int
        状態ベクトルの次元数
    is_discrete : bool
        出力を離散化するかどうか
    hidden_dims : list[int], optional
        隠れ層の次元数のリスト
    """

    def __init__(
        self, state_dim: int, is_discrete: bool, hidden_dims: list[int] = [128, 128]
    ):
        super().__init__()
        self.is_discrete = is_discrete
        self.hidden_dims = hidden_dims

        # 入力次元は (state_dim * 3)
        input_dim = state_dim * 3
        layers = []
        for hidden_dim in hidden_dims:
            # 全結合層とReLUを順番に追加
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, state_dim)

    def forward(
        self,
        parent_article_state: torch.Tensor,
        parent_comment_state: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        # 親記事、親コメント、前回の状態を結合
        x = torch.cat(
            [parent_article_state, parent_comment_state, previous_state], dim=-1
        )

        # 隠れ層
        x = self.hidden_layers(x)

        # 出力層
        pred_state = self.output_layer(x)
        # 今回はソフトマックスで確率分布を出力
        pred_state = torch.softmax(pred_state, dim=-1)

        # 離散化指定がある場合は1-hotにする
        if self.is_discrete:
            one_hot = torch.zeros_like(pred_state)
            one_hot[
                torch.arange(pred_state.size(0)), torch.argmax(pred_state, dim=-1)
            ] = 1
            pred_state = one_hot

        return pred_state


def train_and_evaluate(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    num_epochs: int = 1000,
) -> tuple[list[float], list[float]]:
    """

    モデルを訓練し、損失の履歴を返す。

    Parameters
    ----------
    train_loader : DataLoader
        訓練データ
    val_loader : DataLoader
        検証データ
    model : nn.Module
        学習対象のモデル
    num_epochs : int, optional
        学習エポック数

    Returns
    -------
    tuple[list[float], list[float]]
        (train_loss_history, val_loss_history)
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss_history = []
    val_loss_history = []

    # 指定エポック数だけ学習
    for epoch in range(num_epochs):
        model.train()
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

        average_train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(average_train_loss)

        # 検証
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
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

        average_val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(average_val_loss)

    return train_loss_history, val_loss_history


###############################################################################
# Optuna関連の実装
###############################################################################
def create_dataset_for_user(
    df: pd.DataFrame, user_id: int, device: torch.device
) -> TensorDataset:
    """
    特定のuser_idのデータをソートし、TensorDatasetを作成する。

    Parameters
    ----------
    df : pd.DataFrame
        DBから取得したデータをまとめたDataFrame
    user_id : int
        対象ユーザーID
    device : torch.device
        テンソルを配置するデバイス

    Returns
    -------
    TensorDataset
        (親記事ベクトル, 親コメントベクトル, 前回コメントベクトル, 次回コメントベクトル)をまとめたTensorDataset
    """
    user_data = df[df["user_id"] == user_id]
    user_data_sorted = user_data.sort_values(
        by=["normalized_posted_time", "comment_id"], ascending=[True, False]
    )

    def parse_vector(val):
        # Noneの場合は[0,0,0]を返す
        if val is None:
            return [0, 0, 0]
        # 文字列ならJSONパースを試みる
        elif isinstance(val, str):
            return json.loads(val)
        # すでにリスト型ならそのまま返す
        elif isinstance(val, list):
            return val
        # それ以外は[0,0,0]を返す
        else:
            return [0, 0, 0]

    parent_comment_vectors = [
        parse_vector(v) for v in user_data_sorted["parent_comment_content_vector"][:-1]
    ]
    article_vectors = [
        parse_vector(v) for v in user_data_sorted["article_content_vector"][:-1]
    ]
    previous_comment_vectors = [
        parse_vector(v) for v in user_data_sorted["comment_content_vector"][:-1]
    ]
    next_comment_vectors = [
        parse_vector(v) for v in user_data_sorted["comment_content_vector"][1:]
    ]

    dataset = TensorDataset(
        torch.tensor(article_vectors, dtype=torch.float32).to(device),
        torch.tensor(parent_comment_vectors, dtype=torch.float32).to(device),
        torch.tensor(previous_comment_vectors, dtype=torch.float32).to(device),
        torch.tensor(next_comment_vectors, dtype=torch.float32).to(device),
    )
    return dataset


def objective(trial: optuna.trial.Trial) -> float:
    """

    Optunaのobjective関数。隠れ層数と各層の次元をサンプリングし、
    すべてのユーザーに対する検証損失の平均を返す。

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optunaのトライアルオブジェクト

    Returns
    -------
    float
        検証損失の平均値
    """
    # 隠れ層数を1～3層でサンプリング
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # 各層のユニット数を32～256の範囲でサンプリング (step=32)
    hidden_dims = []
    for i in range(n_layers):
        units = trial.suggest_int(f"n_units_l{i}", 32, 256, step=32)
        hidden_dims.append(units)

    # モデルを定義
    model = NNModel(STATE_DIM, IS_DISCRETE, hidden_dims).to(DEVICE)

    # 全ユーザーの検証損失を集める
    val_losses = []
    for user_id in USER_IDS:
        dataset = create_dataset_for_user(df, user_id, DEVICE)
        if len(dataset) < 2:
            # データが少なすぎる場合はスキップ
            continue

        train_dataset, val_dataset = split_dataset(
            dataset, SPLIT_RATIO, SHOULD_SHUFFLE, random_seed=42
        )
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            continue

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        train_loss, val_loss = train_and_evaluate(
            train_loader, val_loader, model, NUM_EPOCHS
        )
        val_losses.append(val_loss[-1])  # 最終エポックの検証損失を保存

    # 全ユーザーの検証損失平均を最適化対象とする
    if len(val_losses) == 0:
        return 999999.9
    return float(torch.tensor(val_losses).mean().item())


###############################################################################
# メイン処理
###############################################################################
if __name__ == "__main__":

    # データベース接続設定例
    DATABASE_CONFIG = {
        "host": "postgresql_db",
        "database": "yahoo_news_modeling_1",
        "user": "kjqw",
        "password": "1122",
        "port": "5432",
    }

    # SQLクエリ (テーブル名はあなたの環境に合わせて書き換える)
    TRAINING_DATA_QUERY = """
        SELECT *
        FROM training_data_vectorized_sentiment
    """

    # DBからデータを取得
    training_data_vectorized_sentiment = execute_query(
        TRAINING_DATA_QUERY, DATABASE_CONFIG
    )

    # カラム名を設定し、DataFrameに変換
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

    # ユーザーごとの出現回数を数え、100以上のユーザーのみを抽出 (例)
    value_counts = df["user_id"].value_counts()
    filtered_users = value_counts[value_counts >= 100]
    USER_IDS = filtered_users.index.tolist()

    # ハイパーパラメータ
    STATE_DIM = 3
    IS_DISCRETE = False
    SHOULD_SHUFFLE = False
    BATCH_SIZE = 8
    NUM_EPOCHS = 20  # サンプルなので短め
    SPLIT_RATIO = 0.8

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optunaのスタディ設定
    study_name = "nn_hyperparam_search"
    study = optuna.create_study(direction="minimize", study_name=study_name)
    # トライアル回数は適宜調整
    study.optimize(objective, n_trials=10)

    # ベストな結果を表示
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # 結果保存用ディレクトリを作成
    time_now = datetime.now().strftime("%Y%m%d%H%M%S")
    results_dir = Path(__file__).parent / f"optuna_results_{time_now}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 全トライアル結果を保存
    all_trials_data = []
    for t in study.trials:
        trial_dict = {
            "trial_id": t.number,
            "value": t.value if t.value is not None else float("inf"),
            "params": t.params,
        }
        all_trials_data.append(trial_dict)

    with open(results_dir / "all_trials.json", "w") as f:
        json.dump(all_trials_data, f, indent=4)

    # ベストトライアルも保存
    best_data = {"value": study.best_trial.value, "params": study.best_trial.params}
    with open(results_dir / "best_trial.json", "w") as f:
        json.dump(best_data, f, indent=4)

    print(f"全トライアル結果を {results_dir} に保存した")

# %%
