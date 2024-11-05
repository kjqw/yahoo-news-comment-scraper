# %%
# インポート
from classes import Nodes

# %%
# シミュレーションパラメータ設定
user_num = 5  # ユーザー数
article_num = 3  # 記事数
state_dim = 4  # 状態ベクトルの次元数
k_max = 10  # シミュレーションの時刻の最大値

# %%
# ノードの生成と初期化
nodes = Nodes()
nodes.generate_random_nodes(
    user_num=user_num, article_num=article_num, state_dim=state_dim, k_max=k_max
)

# %%
# 全ユーザーの状態更新
nodes.update_all_states(k_max=k_max)

# %%
# 結果の表示
for user_id, user_node in nodes.user_nodes.items():
    print(f"ユーザー {user_id} の状態遷移:")
    for k, (state, influence) in user_node.states.items():
        print(f"  時刻 {k}: 状態 = {state}, 影響度 = {influence}")
    print()
# %%
