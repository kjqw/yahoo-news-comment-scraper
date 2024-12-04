# %%
import sys
from pathlib import Path

import optimize
import simulate
import utils
import visualize

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query

# %%
# 定数の定義
user_num = 3  # ユーザー数
article_num = 2  # 記事数
state_dim = 3  # 状態ベクトルの次元数
k_max = 30  # シミュレーションの時刻の最大値
add_noise = True  # ノイズを加えるかどうか
is_discrete = False  # 状態ベクトルが離散値かどうか
identifier = 1  # メタ情報が同じ時に区別するための識別子

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
# シミュレーションを実行
simulate.main(
    user_num,
    article_num,
    state_dim,
    k_max,
    identifier,
    add_noise,
    is_discrete,
    db_config,
    init_sql_path,
)
# %%
# メタデータIDを設定
metadata_id = None
metadata_id = utils.set_matadata_id(db_config, None)

# %%
# 最適化を実行
optimize.main(
    metadata_id,
    db_config,
)
# %%
# 可視化
figs = visualize.main(metadata_id, db_config)
# %%
