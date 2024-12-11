# %%
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

import utils
from db_manager import execute_query


class StateModel(nn.Module):
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
        super(StateModel, self).__init__()
        self.W_p = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_q = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_s = nn.Parameter(torch.randn(state_dim, state_dim))
        self.b = nn.Parameter(torch.randn(state_dim, 1))
        self.is_discrete = is_discrete

    def forward(
        self,
        parent_article_state,
        parent_article_strength,
        parent_comment_state,
        parent_comment_strength,
        previous_state,
    ):
        """
        順伝播の計算を行う。

        Parameters
        ----------
        parent_article_state : torch.Tensor
            親記事の状態。
        parent_article_strength : torch.Tensor
            親記事の影響度。
        parent_comment_state : torch.Tensor
            親コメントの状態。
        parent_comment_strength : torch.Tensor
            親コメントの影響度。
        previous_state : torch.Tensor
            前回の状態。

        Returns
        -------
        torch.Tensor
            予測された次の状態。
        """

        pred_state = torch.tanh(
            self.W_p @ parent_article_state * parent_article_strength
            + self.W_q @ parent_comment_state * parent_comment_strength
            + self.W_s @ previous_state
            + self.b
        )
        if self.is_discrete:
            pred_state = torch.where(
                pred_state > 0.5, 1, torch.where(pred_state < -0.5, -1, 0)
            )

        return pred_state


def format_df(db_config: dict, metadata_id: int) -> pd.DataFrame:
    """
    データベースからデータを取得し、DataFrameに変換する関数。

    Parameters
    ----------
    db_config : dict
        データベースの接続設定。
    metadata_id : int
        メタデータID。

    Returns
    -------
    pd.DataFrame
        取得したデータを含むDataFrame。
    """
    data = execute_query(
        f"""
        SELECT *
        FROM nodes
        WHERE metadata_id = {metadata_id};
        """,
        db_config,
    )

    columns = execute_query(
        f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'nodes'
        ORDER BY ordinal_position;
        """,
        db_config,
    )
    columns = [column[0] for column in columns]

    df = pd.DataFrame(data, columns=columns)
    return df


def generate_training_data(df: pd.DataFrame) -> dict:
    """
    トレーニングデータを生成する関数。

    Parameters
    ----------
    df : pd.DataFrame
        ノードのデータを含むDataFrame。

    Returns
    -------
    dict
        トレーニングデータを含む辞書。
    """
    data = defaultdict(list)
    user_rows = df[df["node_type"] == "user"]

    for row in user_rows.itertuples():
        if row.k > 0:
            parent_ids = row.parent_ids
            parent_ks = row.parent_ks

            parent_article_state, parent_article_strength = _get_parent_state_strength(
                df, parent_ids[0], parent_ks[0]
            )

            if len(parent_ids) == 1:
                parent_comment_state, parent_comment_strength = (
                    np.zeros((row.state_dim, 1)),
                    0,
                )
            else:
                parent_comment_state, parent_comment_strength = (
                    _get_parent_state_strength(df, parent_ids[1], parent_ks[1])
                )

            previous_state = df[
                (df["node_id"] == row.node_id) & (df["k"] == row.k - 1)
            ]["state"].values[0]

            data[row.node_id].append(
                (
                    parent_article_state,
                    parent_article_strength,
                    parent_comment_state,
                    parent_comment_strength,
                    row.state,
                    previous_state,
                )
            )

    return data


def _get_parent_state_strength(df, node_id, k) -> tuple[np.ndarray, float]:
    """
    親ノードの状態と影響度を取得する関数。

    Parameters
    ----------
    df : pd.DataFrame
        ノードのデータを含むDataFrame。
    node_id : int
        親ノードのID。
    k : int
        時刻。

    Returns
    -------
    tuple[np.ndarray, float]
        親ノードの状態と影響度。
    """
    parent_state = df[(df["node_id"] == node_id) & (df["k"] == k)]["state"].values[0]
    parent_strength = df[(df["node_id"] == node_id) & (df["k"] == k)][
        "strength"
    ].values[0]
    return parent_state, parent_strength


def loss_function(pred_state, true_state, is_discrete):
    """
    損失関数を計算する関数。

    Parameters
    ----------
    pred_state : torch.Tensor
        予測された状態。
    true_state : torch.Tensor
        真の状態。
    is_discrete : bool
        離散化するかどうか。

    Returns
    -------
    torch.Tensor
        損失値。
    """
    if is_discrete:
        discrete_pred_state = torch.where(
            pred_state > 0.5, 1, torch.where(pred_state < -0.5, -1, 0)
        )
        loss = torch.sum((true_state - discrete_pred_state) ** 2)
    else:
        loss = torch.sum((true_state - pred_state) ** 2)
    return loss


def optimize_params(
    data: dict,
    state_dim: int,
    user_id: int,
    is_discrete: bool,
    epochs: int = 1000,
    batch_size: int = 32,
):
    """
    パラメータを最適化する関数。

    Parameters
    ----------
    data : dict
        トレーニングデータ。
    state_dim : int
        状態の次元数。
    user_id : int
        ユーザーID。
    is_discrete : bool
        離散化するかどうか。
    epochs : int, optional
        エポック数 (デフォルトは1000)。
    batch_size : int, optional
        バッチサイズ (デフォルトは32)。

    Returns
    -------
    dict
        最適化されたパラメータ。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StateModel(state_dim, is_discrete).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_data = data[user_id]
    parent_article_states = torch.tensor(
        [d[0] for d in training_data], dtype=torch.float32, requires_grad=True
    ).to(device)
    parent_article_strengths = torch.tensor(
        [d[1] for d in training_data], dtype=torch.float32, requires_grad=True
    ).to(device)
    parent_comment_states = torch.tensor(
        [d[2] for d in training_data], dtype=torch.float32, requires_grad=True
    ).to(device)
    parent_comment_strengths = torch.tensor(
        [d[3] for d in training_data], dtype=torch.float32, requires_grad=True
    ).to(device)
    true_states = torch.tensor(
        [d[4] for d in training_data], dtype=torch.float32, requires_grad=True
    ).to(device)
    previous_states = torch.tensor(
        [d[5] for d in training_data], dtype=torch.float32, requires_grad=True
    ).to(device)

    dataset = TensorDataset(
        parent_article_states,
        parent_article_strengths,
        parent_comment_states,
        parent_comment_strengths,
        true_states,
        previous_states,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            (
                parent_article_state,
                parent_article_strength,
                parent_comment_state,
                parent_comment_strength,
                true_state,
                previous_state,
            ) = batch
            pred_state = model(
                parent_article_state,
                parent_article_strength,
                parent_comment_state,
                parent_comment_strength,
                previous_state,
            )
            loss = loss_function(pred_state, true_state, is_discrete)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return model.parameters()


def save_params(
    params, state_dim: int, user_id: int, metadata_id: int, db_config: dict
):
    """
    パラメータをデータベースに保存する関数。

    Parameters
    ----------
    params : dict
        最適化されたパラメータ。
    state_dim : int
        状態の次元数。
    user_id : int
        ユーザーID。
    metadata_id : int
        メタデータID。
    db_config : dict
        データベースの接続設定。
    """
    W_p, W_q, W_s, b = params
    W_p_str = utils.ndarray_to_ARRAY(W_p.cpu().detach().numpy())
    W_q_str = utils.ndarray_to_ARRAY(W_q.cpu().detach().numpy())
    W_s_str = utils.ndarray_to_ARRAY(W_s.cpu().detach().numpy())
    b_str = utils.ndarray_to_ARRAY(b.cpu().detach().numpy())

    query = f"""
    INSERT INTO params (node_id, metadata_id, w_p_est, w_q_est, w_s_est, b_est)
    VALUES ({user_id}, {metadata_id}, {W_p_str}, {W_q_str}, {W_s_str}, {b_str})
    """
    execute_query(query, db_config, commit=True)

    query = f"""
    UPDATE params
    SET w_p_true = nodes.w_p,
        w_q_true = nodes.w_q,
        w_s_true = nodes.w_s,
        b_true = nodes.b
    FROM nodes
    WHERE params.node_id = nodes.node_id
    AND nodes.node_id = {user_id}
    """
    execute_query(query, db_config, commit=True)


def main(metadata_id: int, db_config: dict):
    """
    メイン関数。

    Parameters
    ----------
    metadata_id : int
        メタデータID。
    db_config : dict
        データベースの接続設定。
    """
    article_num, user_num, state_dim, is_discrete = execute_query(
        f"""
        SELECT article_num, user_num, state_dim, is_discrete
        FROM metadata
        WHERE metadata_id = {metadata_id}
        """,
        db_config,
    )[0]

    df_data = format_df(db_config, metadata_id)
    training_data = generate_training_data(df_data)

    for user_id in range(article_num, article_num + user_num):
        params = optimize_params(training_data, state_dim, user_id, is_discrete)
        save_params(params, state_dim, user_id, metadata_id, db_config)


if __name__ == "__main__":
    db_config = {
        "host": "postgresql_db",
        "database": "test_db",
        "user": "kjqw",
        "password": "1122",
        "port": "5432",
    }

    metadata_id = None
    metadata_id = utils.set_matadata_id(db_config, metadata_id)
    main(metadata_id, db_config)
