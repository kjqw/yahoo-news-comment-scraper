from __future__ import annotations

import random

import numpy as np

# パラメータの設定
INITIAL_STATE_MIN, INITIAL_STATE_MAX = -1, 1
NOISE_MEAN, NOISE_STD = 0, 0.1
WEIGHT_MIN, WEIGHT_MAX = -1, 1
BIAS_MIN, BIAS_MAX = -1, 1
STRENGTH_MIN, STRENGTH_MAX = 0, 1


class Node:
    """
    各時刻のノードの状態と親ノードを保持するクラス。
    """

    def __init__(self, id: int, state_dim: int):
        self.id = id
        self.state_dim = state_dim
        self.k = 0
        self.parents = {}  # 各時刻における親ノード
        self.children = {}  # 各時刻における子ノード
        self.states = {self.k: self.generate_random_state()}  # 状態ベクトル
        self.strengths = {self.k: self.generate_random_strength()}  # 影響度

    def generate_random_state(self) -> np.ndarray:
        """
        状態ベクトルと影響度を初期化するメソッド
        """
        return np.random.uniform(
            INITIAL_STATE_MIN, INITIAL_STATE_MAX, (self.state_dim, 1)
        )

    def generate_random_strength(self) -> float:
        return np.random.uniform(STRENGTH_MIN, STRENGTH_MAX)

    def add_parent(self, k: int, parent_node: Node | None):
        if k not in self.parents:
            self.parents[k] = []
        self.parents[k].append(parent_node)

    def add_child(self, k: int, child_node: Node | None):
        if k not in self.children:
            self.children[k] = []
        self.children[k].append(child_node)


class UserCommentNode(Node):
    """
    ユーザーコメントノード。親ノードから状態を受け取り、状態を更新する。
    """

    def __init__(self, id: int, state_dim: int):
        super().__init__(id, state_dim)
        self.weights = self.generate_random_weights()
        self.bias = self.generate_random_bias()

    def generate_random_weights(self) -> dict[str, np.ndarray]:
        return {
            "W_p": np.random.uniform(
                WEIGHT_MIN, WEIGHT_MAX, (self.state_dim, self.state_dim)
            ),
            "W_q": np.random.uniform(
                WEIGHT_MIN, WEIGHT_MAX, (self.state_dim, self.state_dim)
            ),
            "W_s": np.random.uniform(
                WEIGHT_MIN, WEIGHT_MAX, (self.state_dim, self.state_dim)
            ),
        }

    def generate_random_bias(self) -> np.ndarray:
        return np.random.uniform(BIAS_MIN, BIAS_MAX, (self.state_dim, 1))

    def update_state(
        self,
        state_parent_article: np.ndarray,
        state_parent_comment: np.ndarray,
        strength_article: float,
        strength_comment: float,
        add_noise: bool = True,
    ) -> None:
        """
        ユーザーコメントノードの状態を更新するメソッド。q_features がない、つまり記事だけを見て更新される場合は strength_comment=0 として対応する。
        """
        # ノイズを加えた状態ベクトル
        if add_noise:
            noise = np.random.normal(NOISE_MEAN, NOISE_STD, (self.state_dim, 1))
        else:
            noise = np.zeros((self.state_dim, 1))
        new_state = (
            self.weights["W_p"] @ state_parent_article * strength_article
            + self.weights["W_q"] @ state_parent_comment * strength_comment
            + self.weights["W_s"] @ self.states[self.k]
            + self.bias
            + noise
        )

        # 状態ベクトルと影響度を更新
        self.k += 1
        self.states[self.k] = new_state
        self.strengths[self.k] = self.generate_random_strength()


class ArticleNode(Node):
    """
    記事ノード。親を持たず、状態は固定。
    """

    def __init__(self, id: int, state_dim: int, kmax: int):
        super().__init__(id, state_dim)
        for k in range(1, kmax + 1):
            self.states[k] = self.states[0]
            self.strengths[k] = self.generate_random_strength()


class Nodes:
    """
    全ノードを保持し、親ノードや状態遷移を管理するクラス。
    """

    def __init__(self, article_num: int, user_num: int):
        self.article_num = article_num
        self.user_num = user_num
        self.user_nodes = {}
        self.article_nodes = {}

    def generate_random_nodes(self, state_dim: int, k_max: int) -> None:
        """
        時刻 k_max までランダムに親子関係を設定するメソッド。
        """
        # 記事ノードの生成
        for article_id in range(self.article_num):
            article_node = ArticleNode(id=article_id, state_dim=state_dim, kmax=k_max)
            self.article_nodes[article_id] = article_node

        # ユーザーコメントノードの生成
        for user_id in range(self.article_num, self.article_num + self.user_num):
            user_node = UserCommentNode(id=user_id, state_dim=state_dim)
            self.user_nodes[user_id] = user_node

            # 親子関係の設定
            for k in range(1, k_max + 1):
                # 親記事ノードをランダムに選択
                parent_article_id = random.randint(0, self.article_num - 1)
                parent_article_node = self.article_nodes[parent_article_id]
                user_node.add_parent(k, parent_article_node)
                parent_article_node.add_child(k, user_node)

                # 親ユーザーコメントノードをランダムに選択(None の場合もある)
                parent_user_id = random.choice(
                    list(range(self.article_num, user_id)) + [None]
                )
                if parent_user_id is not None:
                    parent_user_node = self.user_nodes[parent_user_id]
                    user_node.add_parent(k, parent_user_node)
                    parent_user_node.add_child(k, user_node)

        # kでソート
        self.user_nodes = dict(sorted(self.user_nodes.items(), key=lambda x: x[0]))
        self.article_nodes = dict(
            sorted(self.article_nodes.items(), key=lambda x: x[0])
        )

    def update_all_states(self) -> None:
        """
        全ユーザーの状態を更新するメソッド。
        """
        for user_id, user_node in self.user_nodes.items():
            # 親ノードから状態を受け取り、状態を更新
            for k, parent_nodes in user_node.parents.items():
                state_parent_article = np.zeros((user_node.state_dim, 1))
                strength_article = 0
                state_parent_comment = np.zeros((user_node.state_dim, 1))
                strength_comment = 0
                for parent_node in parent_nodes:
                    if isinstance(parent_node, ArticleNode) and k > 0:
                        state_parent_article = parent_node.states[k - 1]
                        strength_article = parent_node.strengths[k - 1]
                    elif isinstance(parent_node, UserCommentNode) and k > 0:
                        state_parent_comment = parent_node.states[k - 1]
                        strength_comment = parent_node.strengths[k - 1]
                user_node.update_state(
                    state_parent_article,
                    state_parent_comment,
                    strength_article,
                    strength_comment,
                )
