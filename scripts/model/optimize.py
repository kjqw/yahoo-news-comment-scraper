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
db_config = {
    "host": "postgresql_db",
    "database": "test_db",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}
# %%
query = """
SELECT * FROM nodes;
"""
data = db_manager.execute_query(query, db_config)
# %%
query = """
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'nodes'
ORDER BY ordinal_position;
"""
columns = db_manager.execute_query(query, db_config)
columns = [column[0] for column in columns]
# %%
columns
# %%
df_data = pd.DataFrame(data, columns=columns)
# %%
df_data


# %%
def generate_training_data(df: pd.DataFrame):
    data = defaultdict(list)
    user_rows = df[df["node_type"] == "user"]
    for row in user_rows.itertuples():
        if row.k > 0:
            parent_ids = row.parent_ids
            parent_ks = row.parent_ks
            parent_article_state = df[
                (df["node_id"] == parent_ids[0]) & (df["k"] == parent_ks[0])
            ]["state"].values[0]
            parent_article_strength = df[
                (df["node_id"] == parent_ids[0]) & (df["k"] == parent_ks[0])
            ]["strength"].values[0]

            if len(parent_ids) == 1:
                parent_comment_state = np.zeros((row.state_dim, row.state_dim))
                parent_comment_strength = 0
            else:
                parent_comment_state = df[
                    (df["node_id"] == parent_ids[1]) & (df["k"] == parent_ks[1])
                ]["state"].values[0]
                parent_comment_strength = df[
                    (df["node_id"] == parent_ids[1]) & (df["k"] == parent_ks[1])
                ]["strength"].values[0]

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


training_data = generate_training_data(df_data)

# %%
training_data[4][0]
# %%
article_num = df_data[df_data["node_type"] == "article"]["node_id"].nunique()
user_num = df_data[df_data["node_type"] == "user"]["node_id"].nunique()
state_dim = df_data["state_dim"].values[0]
k_max = df_data["k"].max()


# %%
def loss_function(params, data, user_id):
    W_p = params[: state_dim**2].reshape(state_dim, state_dim)
    W_q = params[state_dim**2 : 2 * state_dim**2].reshape(state_dim, state_dim)
    W_s = params[2 * state_dim**2 : 3 * state_dim**2].reshape(state_dim, state_dim)
    b = params[3 * state_dim**2 : 3 * state_dim**2 + state_dim]

    loss = 0
    for (
        parent_article_state,
        parent_article_strength,
        parent_comment_state,
        parent_comment_strength,
        state,
        previous_state,
    ) in data[user_id]:
        pred_state = (
            W_p @ parent_article_state * parent_article_strength
            + W_q @ parent_comment_state * parent_comment_strength
            + W_s @ previous_state
            + b
        )
        loss += np.linalg.norm(state - pred_state)

    return loss


# %%
def optimize_params(data, user_id, epochs: int = 5):
    initial_params = np.random.rand(3 * state_dim**2 + state_dim)

    for epoch in range(epochs):
        res = minimize(loss_function, initial_params, args=(data, user_id))
        initial_params = res.x

    return initial_params


# %%
user_id = 4
params = optimize_params(training_data, user_id)

# %%
W_p_est = params[: state_dim**2].reshape(state_dim, state_dim)
W_q_est = params[state_dim**2 : 2 * state_dim**2].reshape(state_dim, state_dim)
W_s_est = params[2 * state_dim**2 : 3 * state_dim**2].reshape(state_dim, state_dim)
b_est = params[3 * state_dim**2 : 3 * state_dim**2 + state_dim]

# %%
print(f"{W_p_est=}")
print(np.array(df_data[df_data["node_id"] == user_id]["w_p"].values[0]))

# %%
