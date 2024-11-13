"""
実行前に`test_db`データベースを作成しておく必要がある。
例えば、以下のコマンドで`test_db`データベースを作成できる。
```sh
psql -h postgresql_db -U kjqw -d postgres -c "CREATE DATABASE test_db;"
```
"""

# %%
import sys
from pathlib import Path

import utils

sys.path.append(str(Path(__file__).parents[1]))

import db_manager

# %%
user_num = 5  # ユーザー数
article_num = 4  # 記事数
state_dim = 3  # 状態ベクトルの次元数
k_max = 20  # シミュレーションの時刻の最大値
db_config = {
    "host": "postgresql_db",
    "database": "test_db",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}  # データベースの接続設定
init_sql_path = (
    Path(__file__).parent / "init.sql"
)  # テーブルの初期化用SQLファイルのパス

# %%
# ノードのインスタンスを生成。ランダムに初期値や重み行列が設定される
nodes = utils.Nodes(article_num, user_num, state_dim, k_max)
# ノードの親子関係をランダムに生成
nodes.generate_random_nodes(state_dim)
# 状態ベクトルを更新
nodes.update_all_states()

# %%
# データベースの初期化
with init_sql_path.open() as f:
    init_sql = f.read()
db_manager.execute_query(init_sql, db_config, commit=True)

# %%
# データベースにノードの情報を保存
nodes.save_to_db(db_config)
# %%
