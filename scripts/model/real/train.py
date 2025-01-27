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
from tqdm import tqdm, trange

# データベースモジュールのパスを追加して、既存のexecute_queryを使う
sys.path.append(str(Path(__file__).parents[2]))
from db_manager import execute_query


def parse_vector(val: str | list[float] | None) -> list[float]:
    """
    JSON文字列やリスト、Noneなどを共通的に処理してfloatリストを返す

    Parameters
    ----------
    val : str | list[float] | None
        JSON文字列、すでにリスト型のベクトル、またはNone

    Returns
    -------
    list[float]
    """
    if val is None:
        return [0, 0, 0]
    elif isinstance(val, str):
        return json.loads(val)
    elif isinstance(val, list):
        return val
    else:
        return [0, 0, 0]


def create_dataset_for_user(
    df: pd.DataFrame, user_id: int, device: torch.device
) -> TensorDataset:
    """
    指定したユーザーIDのデータをソートし、TensorDatasetを作成する

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

    article_vecs = [
        parse_vector(v) for v in user_data_sorted["article_content_vector"][:-1]
    ]
    parent_comment_vecs = [
        parse_vector(v) for v in user_data_sorted["parent_comment_content_vector"][:-1]
    ]
    prev_comment_vecs = [
        parse_vector(v) for v in user_data_sorted["comment_content_vector"][:-1]
    ]
    next_comment_vecs = [
        parse_vector(v) for v in user_data_sorted["comment_content_vector"][1:]
    ]

    dataset = TensorDataset(
        torch.tensor(article_vecs, dtype=torch.float32).to(device),
        torch.tensor(parent_comment_vecs, dtype=torch.float32).to(device),
        torch.tensor(prev_comment_vecs, dtype=torch.float32).to(device),
        torch.tensor(next_comment_vecs, dtype=torch.float32).to(device),
    )
    return dataset


def split_dataset(
    dataset: TensorDataset,
    split_ratio: float,
    should_shuffle: bool = True,
    random_seed: int | None = None,
) -> tuple[Subset, Subset]:
    """
    データセットを訓練用と評価用に分割する

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
    親記事、親コメント、前回の状態を入力して次の状態を予測するNN

    Parameters
    ----------
    state_dim : int
        状態ベクトルの次元数
    is_discrete : bool
        出力を離散化するかどうか
    hidden_dims : list[int]
        隠れ層のユニット数リスト
    """

    def __init__(
        self,
        state_dim: int,
        is_discrete: bool,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.is_discrete = is_discrete

        input_dim = state_dim * 3
        layers = []
        for hidden_dim in hidden_dims:
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
        x = torch.cat(
            [parent_article_state, parent_comment_state, previous_state], dim=-1
        )
        x = self.hidden_layers(x)
        pred_state = self.output_layer(x)

        pred_state = torch.softmax(pred_state, dim=-1)
        if self.is_discrete:
            one_hot = torch.zeros_like(pred_state)
            one_hot[
                torch.arange(pred_state.size(0)), torch.argmax(pred_state, dim=-1)
            ] = 1
            pred_state = one_hot

        return pred_state


def train_and_evaluate(
    train_loader: DataLoader, val_loader: DataLoader, model: nn.Module, num_epochs: int
) -> float:
    """
    モデルを訓練し、最終エポックの評価損失を返す

    Parameters
    ----------
    train_loader : DataLoader
        訓練データ
    val_loader : DataLoader
        検証データ
    model : nn.Module
        学習対象のモデル
    num_epochs : int
        学習エポック数

    Returns
    -------
    float
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss_history = []
    val_loss_history = []

    for _ in range(num_epochs):
        model.train()
        train_loss = 0.0
        for p_a, p_c, prev_s, next_s in train_loader:
            optimizer.zero_grad()
            pred_s = model(p_a, p_c, prev_s)
            loss = criterion(pred_s, next_s)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for p_a, p_c, prev_s, next_s in val_loader:
                pred_s = model(p_a, p_c, prev_s)
                loss = criterion(pred_s, next_s)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

    return val_loss_history[-1]  # 最終エポックの検証損失を返す


def objective_single_trial(
    trial: optuna.trial.Trial,
    user_dataset: TensorDataset,
    state_dim: int,
    is_discrete: bool,
    device: torch.device,
    split_ratio: float,
    should_shuffle: bool,
    batch_size: int,
    num_epochs: int,
) -> float:
    """
    1つのトライアルを実行して評価損失を返す

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optunaトライアル
    user_dataset : TensorDataset
        対象ユーザーのデータセット
    state_dim : int
        状態ベクトルの次元数
    is_discrete : bool
        出力を離散化するかどうか
    device : torch.device
        テンソル配置先
    split_ratio : float
        訓練データ割合
    should_shuffle : bool
        分割時にシャッフルするかどうか
    batch_size : int
        バッチサイズ
    num_epochs : int
        学習エポック数

    Returns
    -------
    float
    """
    # 隠れ層数を1～3でサンプリング
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # 各層のユニット数を32～256の範囲でサンプリング (step=32)
    hidden_dims = []
    for i in range(n_layers):
        units = trial.suggest_int(f"n_units_l{i}", 32, 256, step=32)
        hidden_dims.append(units)

    # データセット分割
    train_dataset, val_dataset = split_dataset(
        user_dataset, split_ratio, should_shuffle, random_seed=42
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        return 999999.9  # 無意味な値を返して打ち切り

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # モデルを生成して学習
    model = NNModel(state_dim, is_discrete, hidden_dims).to(device)
    val_loss = train_and_evaluate(train_loader, val_loader, model, num_epochs)

    return val_loss


def optimize_for_user(
    user_id: int,
    df: pd.DataFrame,
    study_name_prefix: str,
    n_trials: int,
    state_dim: int,
    is_discrete: bool,
    device: torch.device,
    split_ratio: float,
    should_shuffle: bool,
    batch_size: int,
    num_epochs: int,
) -> optuna.Study:
    """
    特定のユーザーIDのデータのみでOptuna最適化を行う

    Parameters
    ----------
    user_id : int
        対象ユーザーID
    df : pd.DataFrame
        データフレーム
    study_name_prefix : str
        Study名のプレフィックス
    n_trials : int
        トライアル回数
    state_dim : int
        状態ベクトルの次元数
    is_discrete : bool
        出力を離散化するかどうか
    device : torch.device
        テンソル配置先
    split_ratio : float
        訓練データ割合
    should_shuffle : bool
        データ分割時にシャッフルするかどうか
    batch_size : int
        バッチサイズ
    num_epochs : int
        学習エポック数

    Returns
    -------
    optuna.Study
    """
    user_dataset = create_dataset_for_user(df, user_id, device)
    if len(user_dataset) < 2:
        # データが少なすぎる場合はスキップ用に適当なStudyを返す
        dummy_study = optuna.create_study(direction="minimize")
        return dummy_study

    # Study作成
    study = optuna.create_study(
        direction="minimize", study_name=f"{study_name_prefix}_u{user_id}"
    )

    # tqdmでトライアルを回す
    current_best = float("inf")
    for trial_idx in trange(n_trials, desc=f"User {user_id} Trials"):
        trial = study.ask()
        value = objective_single_trial(
            trial,
            user_dataset,
            state_dim,
            is_discrete,
            device,
            split_ratio,
            should_shuffle,
            batch_size,
            num_epochs,
        )
        study.tell(trial, value)

        # ログを表示
        if value < current_best:
            current_best = value
        print(
            f"  Trial {trial_idx} finished with value: {value:.6f}. Current best: {current_best:.6f}"
        )

    return study


if __name__ == "__main__":

    # データベース接続設定例
    DATABASE_CONFIG = {
        "host": "postgresql_db",
        "database": "yahoo_news_modeling_1",
        "user": "kjqw",
        "password": "1122",
        "port": "5432",
    }

    # クエリ
    TRAINING_DATA_QUERY = """
        SELECT *
        FROM training_data_vectorized_sentiment
    """
    # DBからデータを取得
    training_data = execute_query(TRAINING_DATA_QUERY, DATABASE_CONFIG)

    # カラム名
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
    df = pd.DataFrame(training_data, columns=COLUMN_NAMES)

    # ユーザーごとの出現回数を数え、100件以上のユーザーのみを抽出 (例)
    value_counts = df["user_id"].value_counts()
    filtered_users = value_counts[value_counts >= 100]
    USER_IDS = filtered_users.index.tolist()

    # 実験用ハイパーパラメータ
    STATE_DIM = 3
    IS_DISCRETE = False
    SPLIT_RATIO = 0.8
    SHOULD_SHUFFLE = False
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    N_TRIALS = 10  # 各ユーザーあたりのトライアル数

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time_now = datetime.now().strftime("%Y%m%d%H%M%S")
    results_path = Path(__file__).parent / f"data/optuna_results_{time_now}"
    results_path.mkdir(parents=True, exist_ok=True)

    # 全ユーザーをループして個別最適化
    for user_id in tqdm(USER_IDS, desc="Optimize per user"):
        study = optimize_for_user(
            user_id=user_id,
            df=df,
            study_name_prefix="nn_hyperparam_search",
            n_trials=N_TRIALS,
            state_dim=STATE_DIM,
            is_discrete=IS_DISCRETE,
            device=DEVICE,
            split_ratio=SPLIT_RATIO,
            should_shuffle=SHOULD_SHUFFLE,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
        )

        # best trialを表示
        if len(study.trials) == 0:
            print(f"User {user_id} skipped (no valid data).")
            continue
        print(f"User {user_id} best trial:")
        print(f"  Value: {study.best_trial.value}")
        print("  Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        # 結果をJSONに保存
        user_result_path = results_path / f"user_{user_id}"
        user_result_path.mkdir(parents=True, exist_ok=True)

        all_trials_data = []
        for t in study.trials:
            trial_dict = {
                "trial_id": t.number,
                "value": t.value if t.value is not None else float("inf"),
                "params": t.params,
            }
            all_trials_data.append(trial_dict)

        with open(user_result_path / "all_trials.json", "w") as f:
            json.dump(all_trials_data, f, indent=4)

        best_data = {"value": study.best_trial.value, "params": study.best_trial.params}
        with open(user_result_path / "best_trial.json", "w") as f:
            json.dump(best_data, f, indent=4)

    print(f"すべてのユーザーのOptuna結果を {results_path} に保存した")

# %%
