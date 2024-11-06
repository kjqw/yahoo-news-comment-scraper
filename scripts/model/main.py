# %%
# インポート
from classes import Nodes

# %%
# シミュレーションパラメータ設定
user_num = 5  # ユーザー数
article_num = 2  # 記事数
state_dim = 3  # 状態ベクトルの次元数
k_max = 4  # シミュレーションの時刻の最大値

# %%
# ノードの生成
nodes = Nodes(article_num, user_num)
nodes.generate_random_nodes(state_dim, k_max)
# %%
nodes.__dict__
# %%
nodes.user_nodes[2].states
# %%
nodes.update_all_states()

# %%
nodes.user_nodes[2].states
# %%
