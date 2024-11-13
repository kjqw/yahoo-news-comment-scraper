# %%
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.append(str(Path(__file__).parents[1]))

import db_manager

# %%
# データベースの接続設定を指定
db_config = {
    "host": "postgresql_db",
    "database": "test_db",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}

# %%
# データベースから"nodes"テーブルの全データを取得するクエリ
query = "SELECT * FROM nodes;"

# クエリを実行してデータを取得
data = db_manager.execute_query(query, db_config)

# %%
# "nodes"テーブルのカラム名を取得するクエリ
query = """
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'nodes'
ORDER BY ordinal_position;
"""

# クエリを実行してカラム名のリストを取得
columns = db_manager.execute_query(query, db_config)

# カラム名を整える
columns = [column[0] for column in columns]

# %%
# 取得したカラム名を確認
columns

# %%
# データをデータフレームとして読み込む
df_data = pd.DataFrame(data, columns=columns)

# %%
# データフレームの内容を表示
df_data


# %%
def generate_training_data(df: pd.DataFrame) -> dict:
    """
    訓練データを生成する関数。

    Parameters
    ----------
    df : pd.DataFrame
        データベースから取得したデータフレーム。

    Returns
    -------
    dict
        各ユーザーごとの訓練データを保持する辞書。
    """
    # デフォルト辞書を作成
    data = defaultdict(list)

    # "user"タイプのノードを抽出
    user_rows = df[df["node_type"] == "user"]

    # 各ユーザーノードの行についてループ処理
    for row in user_rows.itertuples():
        # kが0のときは親ノードが存在しないためスキップ
        if row.k > 0:
            # 親ノードIDとk値を取得
            parent_ids = row.parent_ids
            parent_ks = row.parent_ks

            # 記事ノードの状態と強度を取得
            parent_article_state, parent_article_strength = _get_parent_state_strength(
                df, parent_ids[0], parent_ks[0]
            )

            # コメントノードが存在しない場合、ゼロ行列と0を設定
            if len(parent_ids) == 1:
                parent_comment_state, parent_comment_strength = (
                    np.zeros((row.state_dim, row.state_dim)),
                    0,
                )
            else:
                # コメントノードの状態と強度を取得
                parent_comment_state, parent_comment_strength = (
                    _get_parent_state_strength(df, parent_ids[1], parent_ks[1])
                )

            # 前のステップの状態を取得
            previous_state = df[
                (df["node_id"] == row.node_id) & (df["k"] == row.k - 1)
            ]["state"].values[0]

            # 各データを辞書に追加
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
    親ノードの状態と強度を取得するヘルパー関数。

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム。
    node_id : int
        親ノードのID。
    k : int
        ノードのk値。

    Returns
    -------
    tuple
        親ノードの状態と強度。
    """
    # 指定したノードIDとk値に対応する状態と強度を取得
    parent_state = df[(df["node_id"] == node_id) & (df["k"] == k)]["state"].values[0]
    parent_strength = df[(df["node_id"] == node_id) & (df["k"] == k)][
        "strength"
    ].values[0]
    return parent_state, parent_strength


# 訓練データを生成
training_data = generate_training_data(df_data)

# %%
# 特定ユーザーの訓練データの一部を表示
training_data[4][0]

# %%
# 記事数、ユーザー数、状態の次元、および最大k値を取得
article_num = df_data[df_data["node_type"] == "article"]["node_id"].nunique()
user_num = df_data[df_data["node_type"] == "user"]["node_id"].nunique()
state_dim = df_data["state_dim"].values[0]
k_max = df_data["k"].max()


# %%
def loss_function(params: np.ndarray, data: dict, user_id: int) -> float:
    """
    損失関数を計算する。

    Parameters
    ----------
    params : np.ndarray
        最適化するパラメータ。
    data : dict
        訓練データ。
    user_id : int
        ユーザーID。

    Returns
    -------
    float
        損失の値。
    """
    # パラメータをリシェイプして取得
    W_p, W_q, W_s, b = _reshape_params(params)

    # 損失を初期化
    loss = 0

    # 各データポイントについて予測値と実際の値との差分の二乗和を計算
    for (
        parent_article_state,
        parent_article_strength,
        parent_comment_state,
        parent_comment_strength,
        state,
        previous_state,
    ) in data[user_id]:
        # 予測状態を計算
        pred_state = (
            W_p @ parent_article_state * parent_article_strength
            + W_q @ parent_comment_state * parent_comment_strength
            + W_s @ previous_state
            + b
        )
        # ノルムで損失を加算
        loss += np.linalg.norm(state - pred_state)

    return loss


def _reshape_params(
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    パラメータをリシェイプして取得するヘルパー関数。

    Parameters
    ----------
    params : np.ndarray
        フラットなパラメータ配列。

    Returns
    -------
    tuple
        W_p, W_q, W_s, b に分割されたパラメータ。
    """
    # パラメータを行列とベクトルに変換
    W_p = params[: state_dim**2].reshape(state_dim, state_dim)
    W_q = params[state_dim**2 : 2 * state_dim**2].reshape(state_dim, state_dim)
    W_s = params[2 * state_dim**2 : 3 * state_dim**2].reshape(state_dim, state_dim)
    b = params[3 * state_dim**2 : 3 * state_dim**2 + state_dim]
    return W_p, W_q, W_s, b


# %%
def optimize_params(data: dict, user_id: int, epochs: int = 5) -> np.ndarray:
    """
    パラメータの最適化を行う関数。

    Parameters
    ----------
    data : dict
        訓練データ。
    user_id : int
        ユーザーID。
    epochs : int, optional
        エポック数, デフォルトは5。

    Returns
    -------
    np.ndarray
        最適化されたパラメータ。
    """
    # 初期パラメータをランダムに生成
    initial_params = np.random.rand(3 * state_dim**2 + state_dim)

    # エポック数分の最適化を実行
    for _ in range(epochs):
        res = minimize(loss_function, initial_params, args=(data, user_id))
        initial_params = res.x  # 最適化結果を初期パラメータとして更新

    return initial_params


# %%
# 最適化の対象とするユーザーIDを指定し、最適化を実行
user_id = 4
params = optimize_params(training_data, user_id)

# %%
# 最適化されたパラメータを取得し、リシェイプ
W_p_est, W_q_est, W_s_est, b_est = _reshape_params(params)

# %%
# 最適化されたパラメータを表示
print(f"{W_p_est=}")
print(np.array(df_data[df_data["node_id"] == user_id]["w_p"].values[0]))

# %%
