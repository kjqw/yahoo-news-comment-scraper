# %%
import sys
from pathlib import Path

import numpy as np
import utils

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query

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
article_num = execute_query(
    """
    SELECT COUNT(DISTINCT node_id) AS article_num
    FROM nodes
    WHERE node_type = 'article';
    """,
    db_config,
)[0][0]
user_num = execute_query(
    """
    SELECT COUNT(DISTINCT node_id) AS user_num
    FROM nodes
    WHERE node_type = 'user';
    """,
    db_config,
)[0][0]
state_dim = execute_query(
    """
    SELECT state_dim
    FROM nodes
    LIMIT 1
    """,
    db_config,
)[0][0]
k_max = execute_query(
    """
    SELECT k_max
    FROM nodes
    LIMIT 1
    """,
    db_config,
)[0][0]

# %%
weights = [
    utils.get_params(i, db_config) for i in range(article_num, article_num + user_num)
]
# %%
weights

# %%
true_states = {
    i: [
        np.array(j)
        for j in execute_query(
            f"""
        SELECT state
        FROM nodes
        WHERE node_type = 'user' AND node_id = {i}
        """,
            db_config,
        )
    ]
    for i in range(article_num, article_num + user_num)
}

# %%
true_states

# %%
pred_states = {user_id: [true_states[user_id][0]] for user_id in true_states}
for user_id in range(article_num, article_num + user_num):
    W_p = weights[user_id - article_num]["w_p_est"]
    W_q = weights[user_id - article_num]["w_q_est"]
    W_s = weights[user_id - article_num]["w_s_est"]
    b = weights[user_id - article_num]["b_est"]
    for k in range(1, k_max):
        parent_states = utils.get_parent_state_and_strength(user_id, k, db_config)
        if len(parent_states) == 1:
            state_parent_article, strength_parent_article, _ = parent_states[0]
            state_parent_comment, strength_parent_comment = (
                np.zeros_like(state_parent_article),
                0,
            )
        elif len(parent_states) == 2:
            state_parent_article, strength_parent_article, _ = parent_states[0]
            state_parent_comment, strength_parent_comment, _ = parent_states[1]
        else:
            print("親ノードは2つ以上存在することはありません。")
        pred_states[user_id].append(
            utils.update_method(
                pred_states[user_id][-1],
                W_p,
                state_parent_article,
                strength_parent_article,
                W_q,
                state_parent_comment,
                strength_parent_comment,
                W_s,
                b,
                state_dim,
                add_noise=False,
            )
        )

# %%
for true_state, pred_state in zip(true_states[5], pred_states[5]):
    print(np.hstack([true_state[0], pred_state[0]]))

# %%
