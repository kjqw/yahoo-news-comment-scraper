from __future__ import annotations

import numpy as np
import scipy as sp

INITIAL_STATE_MIN, INITIAL_STATE_MAX = -1, 1
NOISE_MEAN, NOISE_STD = 0, 0.1
WEIGHT_MIN, WEIGHT_MAX = -1, 1
BIAS_MIN, BIAS_MAX = -1, 1
STRENGTH_MIN, STRENGTH_MAX = 0, 1


class Node:
    """
    各時刻のノードの状態と親ノードを保持するクラス

    Attributes
    ----------
    id : int
        ノードのID
    parents : dict[int, list[Node | None]]
        各時刻における親ノードを保持する辞書。keyは時刻、valueは親ノードのクラスのリスト。
    children : dict[int, list[Node | None]]
        各時刻における子ノードを保持する辞書。keyは時刻、valueは子ノードのクラスのリスト。
    state_dim : int
        ノードの状態の次元
    k : int
        現在の時刻。ノードごとに独立しているため、違うノード同士でのkの比較は意味を持たない。
    states : dict[int, list[np.ndarray, float]]
        各時刻のノードの状態。keyは時刻、valueはノードの状態と影響度のリスト。
    is_random : bool
        ノードの状態がランダム変数かどうかのフラグ
    """

    def __init__(
        self,
        id: int,
        parents: dict[int, list[Node | None]],
        state_dim: int,
        is_random: bool = True,
        states: dict[int, list[np.ndarray, float]] | None = None,
    ):
        self.id = id
        self.state_dim = state_dim
        self.parents = parents
        self.k = 0
        self.is_random = is_random

        if is_random:
            random_state = np.random.uniform(
                INITIAL_STATE_MIN, INITIAL_STATE_MAX, (state_dim, 1)
            )
            random_strength = np.random.uniform(STRENGTH_MIN, STRENGTH_MAX)
            self.states = {0: [random_state, random_strength]}
        else:
            self.states = states


class UserCommentNode(Node):
    """
    ユーザのノードの状態と親ノードを保持するクラス

    Attributes
    ----------
    W_s : np.ndarray
        ユーザのノードの状態に対する自己重み行列
    W_p : np.ndarray
        親ノードの状態に対する重み行列
    b : np.ndarray
        バイアス

    Methods
    -------

    """

    def __init__(
        self,
        id: int,
        parents: dict[int, list[UserCommentNode | ArticleNode | None]],
        state_dim: int,
        is_random: bool = True,
        states: dict[int, list[np.ndarray, float]] | None = None,
        W_s: np.ndarray | None = None,
        W_p: np.ndarray | None = None,
        W_q: np.ndarray | None = None,
        b: np.ndarray | None = None,
    ):
        super().__init__(id, parents, state_dim, is_random, states)

        if is_random:
            # self.W_s = np.random.normal(0, 1, (state_dim, state_dim))
            # self.W_p = np.random.normal(0, 1, (state_dim, state_dim))
            # self.b = np.random.normal(0, 1, (state_dim, 1))
            # self.strengths = {0: np.random.normal(0, 1)}

            self.W_s = np.random.uniform(WEIGHT_MIN, WEIGHT_MAX, (state_dim, state_dim))
            self.W_p = np.random.uniform(WEIGHT_MIN, WEIGHT_MAX, (state_dim, state_dim))
            self.W_q = np.random.uniform(WEIGHT_MIN, WEIGHT_MAX, (state_dim, state_dim))
            self.b = np.random.uniform(BIAS_MIN, BIAS_MAX, (state_dim, 1))

        else:
            self.W_s = W_s
            self.W_p = W_p
            self.W_q = W_q
            self.b = b

    def update_state(
        self,
        add_noise: bool = False,
    ) -> np.ndarray:
        """
        ユーザのノードの状態を更新する
        """
        current_state = self.states[self.k]
        for parent in self.parents[self.k].values():
            if type(parent) == UserCommentNode:
                parent_comment_state, parent_comment_strength = parent.states[self.k]
            elif type(parent) == ArticleNode:
                parent_article_state, parent_article_strength = parent.states[self.k]

        new_state = (
            self.W_p @ parent_article_state * parent_article_strength
            + self.W_q @ parent_comment_state * parent_comment_strength
            + self.W_s @ current_state
            + self.b
        )

        if add_noise:
            new_state += np.random.normal(NOISE_MEAN, NOISE_STD, (self.state_dim, 1))

        self.k += 1
        self.states[self.k] = [
            new_state,
            self.states[self.k - 1][1],
        ]  # TODO strengthの更新は後で考える

        return new_state


class ArticleNode(Node):
    pass
