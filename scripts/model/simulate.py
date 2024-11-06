"""
これを実行する前に
```sh
psql -h postgresql_db -U kjqw -d yahoo_news
```
で既存のデータベースに接続し
```sql
CREATE DATABASE test_db;
```
で新しいデータベースを作成しておく
"""

# %%
# インポート
import sys
from pathlib import Path

from classes import Nodes

sys.path.append(str(Path(__file__).parents[1]))

import db_manager

# %%
# シミュレーションパラメータ設定
user_num = 10  # ユーザー数
article_num = 4  # 記事数
state_dim = 5  # 状態ベクトルの次元数
k_max = 100  # シミュレーションの時刻の最大値
db_config = {
    "host": "postgresql_db",
    "database": "test_db",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}

# %%
# # すでにデータがある場合は削除
# sql_path = Path(__file__).parent / "delete_all_table.sql"
# with sql_path.open() as f:
#     query = f.read()
#     db_manager.execute_query(query, db_config=db_config, commit=True)

# %%
# データベースの初期化
init_sql_path = Path(__file__).parent / "init.sql"
with init_sql_path.open() as f:
    query = f.read()
    db_manager.execute_query(query, db_config=db_config, commit=True)

# %%
# ノードの生成
nodes = Nodes(article_num, user_num, state_dim, k_max)
# %%
# nodes.load_from_db(db_config)
# %%
nodes.generate_random_nodes(state_dim)
nodes.update_all_states()

# %%
nodes.save_to_db(db_config)
# %%
