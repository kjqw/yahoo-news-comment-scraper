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
        self.states = {
            self.k: self._initialize_state_with_strength()
        }  # 状態ベクトルと影響度のペア
        self.weights = self._initialize_weights()
        self.bias = self._initialize_bias()

    def _initialize_state_with_strength(self) -> tuple[np.ndarray, float]:
        """
        状態ベクトルと影響度を初期化するメソッド
        """
        state = np.random.uniform(INITIAL_STATE_MIN, INITIAL_STATE_MAX, self.state_dim)
        strength = random.uniform(STRENGTH_MIN, STRENGTH_MAX)
        return state, strength

    def _initialize_weights(self) -> dict[str, np.ndarray]:
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

    def _initialize_bias(self) -> np.ndarray:
        return np.random.uniform(BIAS_MIN, BIAS_MAX, self.state_dim)

    def add_parent(self, k: int, parent_node: Node | None):
        if k not in self.parents:
            self.parents[k] = []
        self.parents[k].append(parent_node)

    def update_state(
        self,
        k: int,
        p_features: np.ndarray,
        q_features: np.ndarray,
        alpha: float,
        beta: float,
    ):
        s_prev = (
            self.states[k - 1][0]
            if k - 1 in self.states
            else self._initialize_state_with_strength()[0]
        )
        self.states[k] = (
            self.weights["W_p"] @ (alpha * p_features)
            + self.weights["W_q"] @ (beta * q_features)
            + self.weights["W_s"] @ s_prev
            + self.bias,
            alpha * beta,  # 状態更新の影響度
        )


class UserCommentNode(Node):
    """
    ユーザーコメントノード。
    """

    pass


class ArticleNode(Node):
    """
    記事ノード。親を持たず、状態は固定。
    """

    def __init__(self, id: int, state_dim: int):
        super().__init__(id, state_dim)
        self.states[0] = self._initialize_state_with_strength()  # 最初の状態として固定


class Nodes:
    """
    全ノードを保持し、親ノードや状態遷移を管理するクラス。
    """

    def __init__(self):
        self.user_nodes = {}
        self.article_nodes = {}

    def generate_random_nodes(
        self, user_num: int, article_num: int, state_dim: int, k_max: int
    ) -> None:
        self.user_nodes = {
            i: UserCommentNode(id=i, state_dim=state_dim) for i in range(user_num)
        }
        self.article_nodes = {
            i: ArticleNode(id=i, state_dim=state_dim)
            for i in range(user_num, user_num + article_num)
        }

        for user_id in range(user_num):
            user_node = self.user_nodes[user_id]
            for k in range(1, k_max + 1):
                parent_nodes = random.sample(
                    list(self.article_nodes.values()) + list(self.user_nodes.values()),
                    k=random.randint(1, 2),
                )
                for parent in parent_nodes:
                    user_node.add_parent(k, parent)

    def update_all_states(self, k_max: int) -> None:
        for k in range(1, k_max + 1):
            for user_id, user_node in self.user_nodes.items():
                # 親ノードから特徴量と影響度を取得
                p_features = np.zeros(user_node.state_dim)
                q_features = np.zeros(user_node.state_dim)
                alpha, beta = 1.0, 1.0  # 初期影響度

                for parent in user_node.parents.get(k, []):
                    if isinstance(parent, ArticleNode):
                        p_features += parent.states[0][0]
                        alpha *= parent.states[0][1]
                    elif isinstance(parent, UserCommentNode):
                        q_features += parent.states.get(
                            k - 1, [np.zeros(user_node.state_dim), 1.0]
                        )[0]
                        beta *= parent.states.get(
                            k - 1, [np.zeros(user_node.state_dim), 1.0]
                        )[1]

                # ユーザーコメントの状態更新
                user_node.update_state(k, p_features, q_features, alpha, beta)
