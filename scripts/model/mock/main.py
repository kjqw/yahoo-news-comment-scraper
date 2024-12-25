"""
実行前に`db_simulate`データベースを作成しておく必要がある。
例えば、以下のコマンドで`db_simulate`データベースを作成できる。
```sh
psql -h postgresql_db -U kjqw -d postgres -c "CREATE DATABASE db_simulate;"
```
"""

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
state_dims = [2, 3, 4, 5]
k_maxs = [10, 30, 50]

user_num = 3  # ユーザー数
article_num = 2  # 記事数
# state_dim = 4  # 状態ベクトルの次元数
# k_max = 30  # シミュレーションの時刻の最大値
add_noise = True  # ノイズを加えるかどうか
is_discrete = True  # 状態ベクトルが離散値かどうか
identifier = 1  # メタ情報が同じ時に区別するための識別子

db_config = {
    "host": "postgresql_db",
    "database": "db_simulate",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}  # データベースの接続設定
init_sql_path = (
    Path(__file__).parent / "init.sql"
)  # テーブルの初期化用SQLファイルのパス
figs_path = (
    Path(__file__).parent / "data/figs_discrete"
    if is_discrete
    else Path(__file__).parent / "data/figs_continuous"
)
figs_path.mkdir(exist_ok=True, parents=True)

for state_dim in state_dims:
    for k_max in k_maxs:
        print(f"state_dim: {state_dim}, k_max: {k_max}")
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
        # メタデータIDを設定
        metadata_id = utils.set_matadata_id(db_config, None)
        # 最適化を実行
        optimize.main(
            metadata_id,
            db_config,
        )
        # 可視化
        figs = visualize.main(metadata_id, db_config)
        # 保存
        for i, fig in enumerate(figs):
            fig_path = figs_path / f"StateDim{state_dim}_KMax{k_max}_User{i}.png"
            fig.savefig(fig_path)
# %%
