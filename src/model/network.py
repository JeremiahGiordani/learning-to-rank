# src/model/network.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Small utilities
# -------------------------

def mlp(dims, act=nn.ReLU, dropout: float = 0.0, last_bn: bool = False) -> nn.Sequential:
    """
    Build a simple MLP with BatchNorm on hidden layers.
    dims: e.g., [in, h1, h2, out]
    """
    layers = []
    for i in range(len(dims) - 1):
        inp, out = dims[i], dims[i + 1]
        is_last = (i == len(dims) - 2)
        layers.append(nn.Linear(inp, out))
        if not is_last:
            layers.append(nn.BatchNorm1d(out))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        elif last_bn:
            layers.append(nn.BatchNorm1d(out))
    return nn.Sequential(*layers)


@dataclass
class RankerConfig:
    node_dim: int
    edge_dim: int
    global_dim: int
    node_hidden: int = 64
    edge_hidden: int = 64
    score_hidden: int = 64
    use_edges: bool = True
    gnn_layers: int = 0         # 0 -> DeepSets+edges only; >0 enables message passing
    dropout: float = 0.0


# -------------------------
# Core Network
# -------------------------

class RankerNet(nn.Module):
    """
    Permutation-equivariant ranker producing a scalar utility u_i per aircraft.

    Forward signature:
        scores = model(node, edge, glob)

        node: (B, N=4, F_n)
        edge: (B, N=4, N=4, F_e)
        glob: (B, F_g)

        returns scores: (B, N=4)

    Architecture:
      - Node encoder: per-aircraft features -> h_i
      - Optional edge encoder + aggregation: messages from j!=i -> m_i
      - Optional message passing (gnn_layers > 0)
      - Global context g = mean_i(concat(h_i, m_i))
      - Score head per node: u_i = MLP([h_i, m_i, g])

    Notes:
      - Equivariance: operations are shared across nodes and use symmetric aggregations (mean).
      - Stable across batches: BatchNorm is applied over node dimension via view/reshape.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        node_hidden: int = 64,
        edge_hidden: int = 64,
        score_hidden: int = 64,
        use_edges: bool = True,
        gnn_layers: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cfg = RankerConfig(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            node_hidden=node_hidden,
            edge_hidden=edge_hidden,
            score_hidden=score_hidden,
            use_edges=use_edges,
            gnn_layers=gnn_layers,
            dropout=dropout,
        )

        # Node encoder
        self.node_enc = mlp([node_dim, node_hidden, node_hidden], dropout=dropout)

        # Edge encoder (shared for i->j), used for edge aggregation and message passing
        self.use_edges = use_edges
        if use_edges:
            self.edge_enc = mlp([edge_dim, edge_hidden, edge_hidden], dropout=dropout)
        else:
            self.edge_enc = None
            edge_hidden = 0  # no edge channels

        # Optional message passing layers
        self.gnn_layers = gnn_layers
        if gnn_layers > 0:
            self.msg_upd = nn.ModuleList()
            for _ in range(gnn_layers):
                # message: combine neighbor node state and encoded edge
                # we keep dimensions fixed (node_hidden)
                self.msg_upd.append(
                    nn.ModuleDict(
                        dict(
                            msg_mlp=mlp([node_hidden + edge_hidden, node_hidden], dropout=dropout),
                            agg_bn=nn.BatchNorm1d(node_hidden),
                            upd_mlp=mlp([node_hidden + node_hidden, node_hidden], dropout=dropout),
                        )
                    )
                )

        # Global context readout
        # Combine current node rep + aggregated edges to form per-node summary, then mean
        per_node_summary_dim = node_hidden + (edge_hidden if use_edges else 0)
        self.global_proj = mlp(
            [per_node_summary_dim, per_node_summary_dim], dropout=dropout
        )  # light projection before mean

        # Score head (shared for all nodes)
        score_in = per_node_summary_dim + global_dim
        self.score_mlp = mlp([score_in, score_hidden, score_hidden, 1], dropout=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # Mildly conservative initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _bn_apply(x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
        """
        Apply BN over the feature channel with a (B*N, C) flatten, then restore shape.
        Accepts x shaped (B,N,C) or (B,N,N,C); BN will be applied to the last dim.
        """
        if x.dim() == 3:
            B, N, C = x.shape
            x2 = x.reshape(B * N, C)
            x2 = bn(x2)
            return x2.view(B, N, C)
        elif x.dim() == 4:
            B, N1, N2, C = x.shape
            x2 = x.reshape(B * N1 * N2, C)
            x2 = bn(x2)
            return x2.view(B, N1, N2, C)
        else:
            raise ValueError("Unsupported tensor rank for BN helper.")

    def forward(self, node: torch.Tensor, edge: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        """
        node: (B, N, F_n)
        edge: (B, N, N, F_e)
        glob: (B, F_g)
        returns scores: (B, N)
        """
        B, N, F_n = node.shape
        assert N == 4, "This implementation assumes N=4 (can be generalized)."

        # Encode nodes: (B,N,Fh)
        h = self.node_enc(node.view(B * N, -1)).view(B, N, -1)  # BN is inside mlp; reshape ensures proper stats

        # Encode edges and aggregate messages: m_i = mean_j Enc(e_ij), j!=i
        if self.use_edges:
            # Zero out self-edges to avoid leakage; then encode
            eye = torch.eye(N, device=edge.device, dtype=torch.bool).unsqueeze(-1)  # (N,N,1)
            e = edge.masked_fill(eye.unsqueeze(0), 0.0)
            e_enc = self.edge_enc(e.view(B * N * N, -1)).view(B, N, N, -1)  # (B,N,N,Feh)

            # mask diag then mean over j
            mask = ~eye  # (N,N)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(edge.dtype)  # (N,1)
            denom = denom.unsqueeze(0)  # (1,N,1)

            m = (e_enc * mask.unsqueeze(0).unsqueeze(-1)).sum(dim=2) / denom  # (B,N,Feh)
        else:
            m = torch.zeros(B, N, 0, device=node.device, dtype=node.dtype)  # empty channel

        # Optional message passing
        if self.gnn_layers > 0 and self.use_edges:
            for layer in self.msg_upd:
                # Build messages from neighbors: concat neighbor h_j with edge e_ij (encoded)
                # We reuse e_enc computed above (depends on initial edge features only).
                # messages_ij: (B,N,N,Fh) from neighbor states + (B,N,N,Feh)
                h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B,N,N,Fh) as j axis
                msg_in = torch.cat([h_j, e_enc], dim=-1)    # (B,N,N,Fh+Feh)
                msg = layer["msg_mlp"](msg_in.view(B * N * N, -1)).view(B, N, N, -1)
                # mask self and mean aggregate over j
                msg = (msg * mask.unsqueeze(0).unsqueeze(-1)).sum(dim=2) / denom  # (B,N,Fh)
                msg = self._bn_apply(msg, layer["agg_bn"])
                # node update (residual style)
                upd_in = torch.cat([h, msg], dim=-1)        # (B,N,2*Fh)
                h = layer["upd_mlp"](upd_in.view(B * N, -1)).view(B, N, -1)

        # Per-node summary used for global context
        if self.use_edges:
            node_sum = torch.cat([h, m], dim=-1)  # (B,N,Fh+Feh)
        else:
            node_sum = h  # (B,N,Fh)

        node_sum_proj = self.global_proj(node_sum.view(B * N, -1)).view(B, N, -1)

        # Global context: symmetric mean over nodes
        g = node_sum_proj.mean(dim=1)  # (B,Fg')  (note: not the same as input glob)
        # Concatenate provided global features to learned global context
        g_all = torch.cat([g, glob], dim=-1)  # (B, Fg'+F_global)

        # Expand global to per-node and score
        g_expand = g_all.unsqueeze(1).expand(-1, N, -1)      # (B,N, Fg'+F_global)
        score_in = torch.cat([node_sum, g_expand], dim=-1)   # (B,N, per_node_summary_dim + global_dim)
        scores = self.score_mlp(score_in.view(B * N, -1)).view(B, N)

        return scores
