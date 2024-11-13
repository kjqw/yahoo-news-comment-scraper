# %%
from pathlib import Path

import classes

# %%
user_num = 5  # ユーザー数
article_num = 4  # 記事数
state_dim = 3  # 状態ベクトルの次元数
k_max = 20  # シミュレーションの時刻の最大値

# %%
nodes = classes.Nodes(article_num, user_num, state_dim, k_max)

# %%
nodes.generate_random_nodes(state_dim)

# %%
nodes.update_all_states()

# # %%
# nodes.save_training_data_to_json(Path(__file__).parent / "data/training_data.json")

# # %%
# nodes.__dict__

# # %%
# nodes.user_nodes[article_num].__dict__

# # %%
# nodes.user_nodes[article_num].states

# %%
import sys

sys.path.append(str(Path(__file__).parents[1]))

import db_manager

# %%
init_sql_path = Path(__file__).parent / "init.sql"
db_config = {
    "host": "postgresql_db",
    "database": "test_db",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}
# %%
with init_sql_path.open() as f:
    init_sql = f.read()
db_manager.execute_query(init_sql, db_config, commit=True)

# %%
nodes.save_to_db(db_config)
# %%
