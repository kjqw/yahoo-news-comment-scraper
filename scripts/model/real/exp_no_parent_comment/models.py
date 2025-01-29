import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.W_p = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_s = nn.Parameter(torch.randn(state_dim, state_dim))
        self.b = nn.Parameter(torch.randn(state_dim, 1))

    def forward(
        self,
        parent_article_state: torch.Tensor,
        previous_state: torch.Tensor,
    ):
        pred_state = (
            torch.matmul(parent_article_state, self.W_p.T)
            + torch.matmul(previous_state, self.W_s.T)
            + self.b.T
        )

        # 出力を softmax で正規化し、確率分布を生成
        pred_state = torch.softmax(pred_state, dim=-1)

        return pred_state


class DiffModel(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.W_p = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_q = nn.Parameter(torch.randn(state_dim, state_dim))
        self.W_s = nn.Parameter(torch.randn(state_dim, state_dim))
        self.b = nn.Parameter(torch.randn(state_dim, 1))

    def forward(
        self,
        parent_article_state: torch.Tensor,
        previous_state: torch.Tensor,
    ):
        pred_state = (
            torch.matmul(parent_article_state - previous_state, self.W_p.T)
            + torch.matmul(previous_state, self.W_s.T)
            + self.b.T
        )

        # 出力を softmax で正規化し、確率分布を生成
        pred_state = torch.softmax(pred_state, dim=-1)

        return pred_state


class NNModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: list[int] = [128, 128]):
        super().__init__()
        self.hidden_dims = hidden_dims

        # 入力次元の初期化
        input_dim = state_dim * 2

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
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        # 入力を結合
        x = torch.cat([parent_article_state, previous_state], dim=-1)

        # 隠れ層の計算
        x = self.hidden_layers(x)

        # 出力層の計算
        pred_state = self.output_layer(x)

        # 出力を softmax で正規化し、確率分布を生成
        pred_state = torch.softmax(pred_state, dim=-1)

        return pred_state
