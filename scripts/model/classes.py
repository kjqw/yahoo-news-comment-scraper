import numpy as np
import scipy as sp


class Node:
    """
    各時刻のノードの状態と親ノードを保持するクラス

    Attributes
    ----------
    id : int
        ノードのID
    parents : dict[int, list[int | None]]
        各時刻における親ノードのIDを保持する辞書。keyは時刻、valueは親ノードのIDのリスト。
    children : dict[int, list[int | None]]
        各時刻における子ノードのIDを保持する辞書。keyは時刻、valueは子ノードのIDのリスト。
    state_dim : int
        ノードの状態の次元
    k : int
        現在の時刻。ノードごとに独立しているため、違うノード同士でのkの比較は意味を持たない。
    states : dict[int, np.ndarray]
        各時刻のノードの状態。keyは時刻、valueはノードの状態で、次元は(k + 1, state_dim)。
    is_random : bool
        ノードの状態がランダム変数かどうかのフラグ
    """

    def __init__(
        self,
        id: int,
        parents: dict[int, list[int | None]],
        state_dim: int,
        is_random: bool = True,
        states: dict[int, np.ndarray] | None = None,
    ):
        self.id = id
        self.state_dim = state_dim
        self.parents = parents
        self.k = 0
        self.is_random = is_random

        if is_random:
            # random_state = np.random.normal(0, 1, (1, state_dim))
            random_state = np.random.uniform(-1, 1, (1, state_dim))
            self.states = {0: random_state}
        else:
            self.states = states


class UserNode(Node):
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
        parents: dict[int, list[int | None]],
        state_dim: int,
        is_random: bool = True,
        states: dict[int, np.ndarray] | None = None,
        W_s: np.ndarray | None = None,
        W_p: np.ndarray | None = None,
        b: np.ndarray | None = None,
        strengths: dict[int, np.ndarray] | None = None,
    ):
        super().__init__(id, parents, state_dim, is_random, states)

        if is_random:
            # self.W_s = np.random.normal(0, 1, (state_dim, state_dim))
            # self.W_p = np.random.normal(0, 1, (state_dim, state_dim))
            # self.b = np.random.normal(0, 1, (state_dim, 1))
            # self.strengths = {0: np.random.normal(0, 1)}

            self.W_s = np.random.uniform(-1, 1, (state_dim, state_dim))
            self.W_p = np.random.uniform(-1, 1, (state_dim, state_dim))
            self.b = np.random.uniform(-1, 1, (state_dim, 1))
            self.strengths = {0: np.random.uniform(0, 1)}

        else:
            self.W_s = W_s
            self.W_p = W_p
            self.b = b
            self.strengths = strengths

    def update_state(
        self,
        current_state: np.ndarray,
        parent_states: np.ndarray,
        parent_strength: np.ndarray,
    ) -> np.ndarray:
        """
        ユーザのノードの状態を更新する

        Parameters
        ----------
        current_state : np.ndarray
            現在のユーザのノードの状態
        parent_states : np.ndarray
            親ノードの状態

        Returns
        -------
        new_state : np.ndarray
            更新後のユーザのノードの状態
        """
        new_state = (
            self.W_p @ parent_states * parent_strength
            + self.W_s @ current_state
            + self.b
        )

        return new_state
