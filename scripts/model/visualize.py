# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
# metadata_id = 2
metadata_id = execute_query(
    """
    SELECT metadata_id
    FROM metadata
    ORDER BY metadata_id DESC
    LIMIT 1
    """,
    db_config,
)[0][0]
article_num, user_num, state_dim, k_max = execute_query(
    f"""
    SELECT article_num, user_num, state_dim, k_max
    FROM metadata
    WHERE metadata_id = {metadata_id}
    """,
    db_config,
)[0]
# %%
article_num, user_num, state_dim, k_max


# %%
weights = [
    utils.get_params(i, db_config) for i in range(article_num, article_num + user_num)
]
# %%
weights

# %%
true_states = {
    i: [
        np.array(j[0])
        for j in execute_query(
            f"""
        SELECT state
        FROM nodes
        WHERE node_type = 'user' AND node_id = {i} AND metadata_id = {metadata_id}
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
    for k in range(1, k_max + 1):
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
true_states[5]
# %%
true_states_plot_data = np.concatenate([i for i in true_states[5]], axis=1)
pred_states_plot_data = np.concatenate([i for i in pred_states[5]], axis=1)
# %%
# プロット設定
sns.set(style="darkgrid")
fig, axes = plt.subplots(state_dim, 1, figsize=(10, state_dim * 2.5), sharex=True)

# 各次元ごとにプロット
for dim, (true_state, pred_state) in enumerate(
    zip(true_states_plot_data, pred_states_plot_data)
):
    # Trueの散布図と折れ線
    sns.scatterplot(
        x=range(k_max + 1),
        y=true_state,
        label="True",
        marker="o",
        color="blue",
        ax=axes[dim],
        legend=False if dim < state_dim - 1 else True,  # 最後のaxのみlegend表示
    )
    sns.lineplot(
        x=range(k_max + 1),
        y=true_state,
        label="True",
        color="blue",
        ax=axes[dim],
        legend=False if dim < state_dim - 1 else True,
    )
    # Predの散布図と折れ線
    sns.scatterplot(
        x=range(k_max + 1),
        y=pred_state,
        label="Pred",
        marker="x",
        color="red",
        ax=axes[dim],
        legend=False if dim < state_dim - 1 else True,
    )
    sns.lineplot(
        x=range(k_max + 1),
        y=pred_state,
        label="Pred",
        color="red",
        linestyle="--",
        ax=axes[dim],
        legend=False if dim < state_dim - 1 else True,
    )

    axes[dim].set_title(f"Dimension {dim + 1}")

# レイアウト調整
plt.tight_layout()
plt.show()

# %%
# 保存
fig_path = Path(__file__).parent / f"data/figs/sample_{metadata_id}.png"
fig_path.parent.mkdir(exist_ok=True, parents=True)
fig.savefig(fig_path)
# %%
