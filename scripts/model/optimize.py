# %%
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import utils
from scipy.optimize import minimize
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query


def set_matadata_id(db_config: dict, metadata_id: int | None = None) -> int:
    """
    メタデータIDを設定する関数。

    Parameters
    ----------
    db_config : dict
        データベースの接続設定。
    metadata_id : int
        メタデータID。

    Returns
    -------
    int
        メタデータID。
    """
    # metadata_idが指定されていない場合、最新のmetadata_idを取得
    if metadata_id is None:
        metadata_id = execute_query(
            """
            SELECT metadata_id
            FROM metadata
            ORDER BY metadata_id DESC
            LIMIT 1
            """,
            db_config,
        )[0][0]

    return metadata_id


# %%
def format_df(db_config: dict) -> pd.DataFrame:
    """
    データベースから取得したデータを整形する関数。

    Parameters
    ----------
    db_config : dict
        データベースの接続設定。

    Returns
    -------
    pd.DataFrame
        整形されたデータフレーム。
    """
    # データベースから"nodes"テーブルの全データを取得する
    data = execute_query(
        f"""
        SELECT *
        FROM nodes
        WHERE metadata_id = {metadata_id};
        """,
        db_config,
    )

    # "nodes"テーブルのカラム名を取得する
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

    # データをデータフレームとして読み込む
    df = pd.DataFrame(data, columns=columns)

    return df


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
        pred_state = np.tanh(
            W_p @ parent_article_state * parent_article_strength
            + W_q @ parent_comment_state * parent_comment_strength
            + W_s @ previous_state
            + b
        )
        discrete_pred_state = np.where(
            pred_state > 0.5, 1, np.where(pred_state < -0.5, -1, 0)
        )
        # ノルムで損失を加算
        loss += np.linalg.norm(state - discrete_pred_state)

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
    b = params[3 * state_dim**2 : 3 * state_dim**2 + state_dim].reshape(state_dim, 1)
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
    for _ in tqdm(range(epochs)):
        res = minimize(loss_function, initial_params, args=(data, user_id))
        initial_params = res.x  # 最適化結果を初期パラメータとして更新

    return initial_params


def save_params(params: np.ndarray, user_id: int, db_config: dict) -> None:
    """
    パラメータをデータベースに保存する関数。

    Parameters
    ----------
    params : np.ndarray
        パラメータ。
    user_id : int
        ユーザーID。
    db_config : dict
        データベースの接続設定。
    """
    # パラメータをリシェイプ
    W_p, W_q, W_s, b = _reshape_params(params)
    W_p_str = utils.ndarray_to_ARRAY(W_p)
    W_q_str = utils.ndarray_to_ARRAY(W_q)
    W_s_str = utils.ndarray_to_ARRAY(W_s)
    b_str = utils.ndarray_to_ARRAY(b)

    # パラメータをデータベースに保存
    query = f"""
    INSERT INTO params (node_id, metadata_id, w_p_est, w_q_est, w_s_est, b_est)
    VALUES ({user_id}, {metadata_id}, {W_p_str}, {W_q_str}, {W_s_str}, {b_str})
    """

    execute_query(
        query,
        db_config,
        commit=True,
    )

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
    execute_query(
        query,
        db_config,
        commit=True,
    )


# %%
if __name__ == "__main__":
    # データベースの接続設定を指定
    db_config = {
        "host": "postgresql_db",
        "database": "test_db",
        "user": "kjqw",
        "password": "1122",
        "port": "5432",
    }

    # メタデータIDを設定
    metadata_id = None
    metadata_id = set_matadata_id(db_config, metadata_id)

    # 訓練データを生成
    df_data = format_df(db_config)
    training_data = generate_training_data(df_data)

    # 記事数、ユーザー数、状態の次元、および最大k値を取得
    article_num, user_num, state_dim, k_max = execute_query(
        f"""
        SELECT article_num, user_num, state_dim, k_max
        FROM metadata
        WHERE metadata_id = {metadata_id}
        """,
        db_config,
    )[0]

    # 最適化の対象とするユーザーIDを指定し、最適化を実行
    for user_id in range(article_num, article_num + user_num):
        params = optimize_params(training_data, user_id)
        save_params(params, user_id, db_config)

    # utils.get_params(article_num, metadata_id, db_config)
# %%
