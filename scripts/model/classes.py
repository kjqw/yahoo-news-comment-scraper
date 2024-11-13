from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parents[1]))

import db_manager

# パラメータ設定
INITIAL_STATE_MIN, INITIAL_STATE_MAX = -1, 1  # 初期状態値の範囲
NOISE_MEAN, NOISE_STD = 0, 0.1  # ノイズの平均と標準偏差
WEIGHT_MIN, WEIGHT_MAX = -1, 1  # 重み行列の範囲
BIAS_MIN, BIAS_MAX = -1, 1  # バイアスベクトルの範囲
STRENGTH_MIN, STRENGTH_MAX = 0, 1  # 影響度の範囲


class Node:
    """
    各時刻におけるノードの状態と親子関係を保持するクラス。
    """

    def __init__(self, id: int, state_dim: int):
        self.id = id
        self.state_dim = state_dim
        self.k = 0  # 現在の時刻
        self.parents = {self.k: []}  # 各時刻における親ノード
        self.states = {}  # 各時刻の状態ベクトル
        self.strengths = {}  # 各時刻の影響度

        # 初期時刻 k = 0 での状態と影響度を初期化
        self.states[self.k] = self.generate_random_state()
        self.strengths[self.k] = self.generate_random_strength()

    def generate_random_state(self) -> np.ndarray:
        """
        状態ベクトルをランダムに初期化するメソッド。
        """
        return np.random.uniform(
            INITIAL_STATE_MIN, INITIAL_STATE_MAX, (self.state_dim, 1)
        )

    def generate_random_strength(self) -> float:
        """
        影響度をランダムに初期化するメソッド。
        """
        return np.random.uniform(STRENGTH_MIN, STRENGTH_MAX)

    def add_parent(self, k_self: int, k_parent: int, parent_node: Node) -> None:
        """
        指定した時刻に親ノードを追加するメソッド。

        Note
        ----
        ArticleNode は親ノードを持たない。
        UserCommentNode は1つのArticleNodeと、0もしくは1つのUserCommentNodeを親ノードとして持つ。
        """
        # k_selfがself.parentsに存在しない場合は初期化
        if k_self not in self.parents:
            self.parents[k_self] = []

        self.parents[k_self].append((k_parent, parent_node))


class UserCommentNode(Node):
    """
    ユーザーコメントノード。親ノードからの状態に基づき、状態を更新する。
    """

    def __init__(self, id: int, state_dim: int):
        super().__init__(id, state_dim)
        self.weights = self.generate_random_weights()  # 状態更新用の重み行列
        self.bias = self.generate_random_bias()  # 状態更新用のバイアスベクトル

    def generate_random_weights(self) -> dict[str, np.ndarray]:
        """
        重み行列をランダムに初期化するメソッド。
        """
        return {
            "W_p": np.random.uniform(
                WEIGHT_MIN, WEIGHT_MAX, (self.state_dim, self.state_dim)
            ),  # 親記事の状態に対応する重み
            "W_q": np.random.uniform(
                WEIGHT_MIN, WEIGHT_MAX, (self.state_dim, self.state_dim)
            ),  # 親コメントの状態に対応する重み
            "W_s": np.random.uniform(
                WEIGHT_MIN, WEIGHT_MAX, (self.state_dim, self.state_dim)
            ),  # 自身の前回状態に対応する重み
        }

    def generate_random_bias(self) -> np.ndarray:
        """
        バイアスベクトルをランダムに初期化するメソッド。
        """
        return np.random.uniform(BIAS_MIN, BIAS_MAX, (self.state_dim, 1))

    def update_state(
        self,
        add_noise: bool = True,
    ) -> None:
        """
        親ノードの状態に基づき、ユーザーコメントノードの状態を更新するメソッド。
        add_noise : bool, Optional
            ノイズを加えるかどうか
        """
        # 指定があればノイズを加える
        noise = (
            np.random.normal(NOISE_MEAN, NOISE_STD, (self.state_dim, 1))
            if add_noise
            else np.zeros((self.state_dim, 1))
        )

        self.k += 1
        # 前のステップの状態を取得
        previous_state = self.states[self.k - 1]

        # 親ノードの状態と影響度を取得
        parent_nodes = self.parents[self.k]
        state_parent_article = np.zeros((self.state_dim, 1))
        strength_article = 0
        state_parent_comment = np.zeros((self.state_dim, 1))
        strength_comment = 0
        for k_parent, parent_node in parent_nodes:
            if isinstance(parent_node, ArticleNode):
                state_parent_article = parent_node.states[k_parent]
                strength_article = parent_node.strengths[k_parent]
            elif isinstance(parent_node, UserCommentNode):
                state_parent_comment = parent_node.states[k_parent]
                strength_comment = parent_node.strengths[k_parent]

        # 状態更新式に基づき新しい状態を計算
        new_state = (
            self.weights["W_p"] @ state_parent_article * strength_article
            + self.weights["W_q"] @ state_parent_comment * strength_comment
            + self.weights["W_s"] @ previous_state
            + self.bias
            + noise
        )

        # 新しい状態と影響度を保存
        self.states[self.k] = new_state
        self.strengths[self.k] = self.generate_random_strength()


class ArticleNode(Node):
    """
    記事ノード。親を持たず、状態は時刻によらず固定。
    """

    def __init__(self, id: int, state_dim: int, k_max: int):
        super().__init__(id, state_dim)
        # 時刻 k=1 から k_max まで状態を固定で保存
        for k in range(1, k_max + 1):
            self.states[k] = self.states[0]
            self.strengths[k] = self.generate_random_strength()


class Nodes:
    """
    全ノードを管理し、親子関係と状態遷移を統括するクラス。
    """

    def __init__(self, article_num: int, user_num: int, state_dim: int, k_max: int):
        self.article_num = article_num
        self.user_num = user_num
        self.state_dim = state_dim
        self.user_nodes = {}  # ユーザーコメントノードの辞書
        self.article_nodes = {}  # 記事ノードの辞書
        self.k_max = k_max

    def generate_random_nodes(self, state_dim: int) -> None:
        """
        ノードをランダムに生成し、時刻 k_max まで親子関係を設定するメソッド。

        Parameters
        ----------
        state_dim : int
            状態ベクトルの次元
        k_max : int
            最大時刻
        """

        # 記事ノードの生成
        for article_id in range(self.article_num):
            article_node = ArticleNode(
                id=article_id, state_dim=state_dim, k_max=self.k_max
            )
            self.article_nodes[article_id] = article_node

        # ユーザーコメントノードの生成
        for user_id in range(self.article_num, self.article_num + self.user_num):
            user_node = UserCommentNode(id=user_id, state_dim=state_dim)
            self.user_nodes[user_id] = user_node

            # 各時刻における親子関係の設定
            for k in range(1, self.k_max + 1):
                # 親記事ノードをランダムに選択
                parent_article_id = random.randint(0, self.article_num - 1)
                parent_article_node = self.article_nodes[parent_article_id]
                user_node.add_parent(
                    k, random.randint(0, self.k_max), parent_article_node
                )

                # 親ユーザーコメントノードの選択（存在しない場合もあり）
                parent_user_ids = list(range(self.article_num, user_id))
                parent_user_id = random.choice(parent_user_ids + [None])
                if parent_user_id is not None:
                    parent_user_node = self.user_nodes[parent_user_id]
                    user_node.add_parent(
                        k, random.randint(0, self.k_max - 1), parent_user_node
                    )

        # ノードを ID 順にソート
        self.user_nodes = dict(sorted(self.user_nodes.items(), key=lambda x: x[0]))
        self.article_nodes = dict(
            sorted(self.article_nodes.items(), key=lambda x: x[0])
        )

    def update_all_states(self) -> None:
        """
        全ユーザーコメントノードの状態を各時刻にわたって更新するメソッド。
        """
        # ID順でユーザーコメントノードを更新し、親が先に更新されるようにする
        for user_id in sorted(self.user_nodes.keys()):
            # 時刻順に状態を更新
            for k in range(1, self.k_max + 1):
                user_node = self.user_nodes[user_id]
                user_node.update_state()

    def generate_training_data(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
        """
        ノードの状態を学習用データに変換するメソッド。
        自分の状態、親の状態・影響度から、重みとバイアスを学習するためのデータを生成する。
        """
        training_data = []
        for user_id, user_node in self.user_nodes.items():
            for k in range(1, self.k_max + 1):
                # 親ノードの状態と影響度を取得
                parent_nodes = user_node.parents[k]
                state_parent_article = np.zeros((self.state_dim, 1))
                strength_article = 0
                state_parent_comment = np.zeros((self.state_dim, 1))
                strength_comment = 0
                for k_parent, parent_node in parent_nodes:
                    if isinstance(parent_node, ArticleNode):
                        state_parent_article = parent_node.states[k_parent]
                        strength_article = parent_node.strengths[k_parent]
                    elif isinstance(parent_node, UserCommentNode):
                        state_parent_comment = parent_node.states[k_parent]
                        strength_comment = parent_node.strengths[k_parent]

                # 学習データを生成
                training_data.append(
                    (
                        state_parent_article,
                        state_parent_comment,
                        user_node.states[k - 1],
                        strength_article,
                        strength_comment,
                    )
                )

        return training_data

    def save_training_data_to_json(self, file_path: Path) -> None:
        """
        学習用データを JSON ファイルに保存するメソッド。
        """
        training_data = self.generate_training_data()
        training_data_json = {
            "metadata": {
                "article_num": self.article_num,
                "user_num": self.user_num,
                "state_dim": self.state_dim,
                "k_max": self.k_max,
            },
            "data": [
                {
                    "state_parent_article": state_parent_article.tolist(),
                    "state_parent_comment": state_parent_comment.tolist(),
                    "state_user_comment": state_user_comment.tolist(),
                    "strength_article": strength_article,
                    "strength_comment": strength_comment,
                }
                for state_parent_article, state_parent_comment, state_user_comment, strength_article, strength_comment in training_data
            ],
        }
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as f:
            json.dump(training_data_json, f, indent=4)
