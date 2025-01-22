import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """
    状態予測モデル。
    親記事、親コメント、前回の状態を入力として次の状態を予測する。

    Parameters
    ----------
    state_dim : int
        状態の次元数。
    is_discrete : bool
        出力を離散化するかどうか。
    """

    def __init__(self, state_dim: int, is_discrete: bool):
        super().__init__()
        # 重み行列とバイアスを定義
        self.W_p = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_q = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_s = nn.Parameter(torch.randn(state_dim, state_dim))
        self.b = nn.Parameter(torch.randn(state_dim, 1))
        self.is_discrete = is_discrete

    def forward(
        self,
        parent_article_state: torch.Tensor,
        parent_comment_state: torch.Tensor,
        previous_state: torch.Tensor,
    ):
        """
        順伝播の計算を行う。

        Parameters
        ----------
        parent_article_state : torch.Tensor
            親記事の状態。
        parent_comment_state : torch.Tensor
            親コメントの状態。Noneだとtorchで扱いにくいので、Noneのときは[2, 2, 2]を入力することにした。
        previous_state : torch.Tensor
            前回の状態。

        Returns
        -------
        torch.Tensor
            予測された次の状態。
        """
        if torch.all(
            parent_comment_state
            == torch.tensor([2, 2, 2], device=parent_comment_state.device)
        ):
            # 親コメントが存在しない場合の状態予測
            pred_state = torch.tanh(
                torch.matmul(parent_article_state, self.W_p.T)
                + torch.matmul(previous_state, self.W_s.T)
                + self.b.T
            )
        else:
            # 親コメントが存在する場合の状態予測
            pred_state = torch.tanh(
                torch.matmul(parent_article_state, self.W_p.T)
                + torch.matmul(parent_comment_state, self.W_q.T)
                + torch.matmul(previous_state, self.W_s.T)
                + self.b.T
            )

        if self.is_discrete:
            # 出力を離散化
            pred_state = torch.where(
                pred_state > 0.5,
                torch.tensor(1.0, dtype=torch.float32, device=pred_state.device),
                torch.where(
                    pred_state < -0.5,
                    torch.tensor(-1.0, dtype=torch.float32, device=pred_state.device),
                    torch.tensor(0.0, dtype=torch.float32, device=pred_state.device),
                ),
            )

        return pred_state


class DiffModel(nn.Module):
    """
    状態予測モデル。
    親記事、親コメント、前回の状態を入力として次の状態を予測する。

    Parameters
    ----------
    state_dim : int
        状態の次元数。
    is_discrete : bool
        出力を離散化するかどうか。
    """

    def __init__(self, state_dim: int, is_discrete: bool):
        super().__init__()
        # 重み行列とバイアスを定義
        self.W_p = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_q = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_s = nn.Parameter(torch.randn(state_dim, state_dim))
        self.b = nn.Parameter(torch.randn(state_dim, 1))
        self.is_discrete = is_discrete

    def forward(
        self,
        parent_article_state: torch.Tensor,
        parent_comment_state: torch.Tensor,
        previous_state: torch.Tensor,
    ):
        """
        順伝播の計算を行う。

        Parameters
        ----------
        parent_article_state : torch.Tensor
            親記事の状態。
        parent_comment_state : torch.Tensor
            親コメントの状態。Noneだとtorchで扱いにくいので、Noneのときは[2, 2, 2]を入力することにした。
        previous_state : torch.Tensor
            前回の状態。

        Returns
        -------
        torch.Tensor
            予測された次の状態。
        """
        if torch.all(
            parent_comment_state
            == torch.tensor([2, 2, 2], device=parent_comment_state.device)
        ):
            # 親コメントが存在しない場合の状態予測
            pred_state = torch.tanh(
                torch.matmul(parent_article_state - previous_state, self.W_p.T)
                + torch.matmul(previous_state, self.W_s.T)
                + self.b.T
            )
        else:
            # 親コメントが存在する場合の状態予測
            pred_state = torch.tanh(
                torch.matmul(parent_article_state - previous_state, self.W_p.T)
                + torch.matmul(parent_comment_state - previous_state, self.W_q.T)
                + torch.matmul(previous_state, self.W_s.T)
                + self.b.T
            )

        if self.is_discrete:
            # 出力を離散化
            pred_state = torch.where(
                pred_state > 0.5,
                torch.tensor(1.0, dtype=torch.float32, device=pred_state.device),
                torch.where(
                    pred_state < -0.5,
                    torch.tensor(-1.0, dtype=torch.float32, device=pred_state.device),
                    torch.tensor(0.0, dtype=torch.float32, device=pred_state.device),
                ),
            )

        return pred_state


class NNModel(nn.Module):
    """
    状態予測モデル。
    親記事、親コメント、前回の状態を入力として次の状態を予測する。

    Parameters
    ----------
    state_dim : int
        入力状態の次元数。
    is_discrete : bool
        出力を離散化するかどうか。
    hidden_dims : list[int], optional
        隠れ層の次元数のリスト。
    """

    def __init__(
        self, state_dim: int, is_discrete: bool, hidden_dims: list[int] = [128, 128]
    ):
        super().__init__()
        self.is_discrete = is_discrete
        self.hidden_dims = hidden_dims

        # 入力次元の初期化
        input_dim = state_dim

        # 隠れ層の定義
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim  # 次の層の入力次元を更新
        self.hidden_layers = nn.Sequential(*layers)

        # 出力層の定義
        self.output_layer = nn.Linear(input_dim, state_dim)

    def forward(
        self,
        parent_article_state: torch.Tensor,
        parent_comment_state: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        順伝播の計算を行う。

        Parameters
        ----------
        parent_article_state : torch.Tensor
            親記事の状態。
        parent_comment_state : torch.Tensor
            親コメントの状態。
        previous_state : torch.Tensor
            前回の状態。

        Returns
        -------
        torch.Tensor
            予測された次の状態。
        """
        # 入力を結合
        x = torch.cat(
            [parent_article_state, parent_comment_state, previous_state], dim=-1
        )

        # 隠れ層の計算
        x = self.hidden_layers(x)

        # 出力層の計算
        pred_state = self.output_layer(x)

        if self.is_discrete:
            # 出力を 1-hot ベクトルに変換
            one_hot = torch.zeros_like(pred_state)
            one_hot[
                torch.arange(pred_state.size(0)), torch.argmax(pred_state, dim=-1)
            ] = 1
            pred_state = one_hot

        return pred_state
