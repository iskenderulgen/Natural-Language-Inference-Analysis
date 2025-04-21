import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomposableAttention(nn.Module):
    """
    PyTorch implementation of the Decomposable Attention model (Parikh et al., 2016).

    Args:
        embedding_matrix (Tensor): pretrained embeddings (vocab_size × embed_dim)
        hidden_size (int): hidden dimensionality for feed-forward nets
        num_classes (int): number of output labels (e.g., 3 for SNLI)
        dropout (float): dropout rate
    """
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Embedding layer (frozen)
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=True,
            padding_idx=0,
        )
        embed_dim = embedding_matrix.size(1)
        self.projection = nn.Linear(embed_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Attend feed-forward F
        self.attend_ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Compare feed-forward G
        self.compare_ff = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Aggregate feed-forward H
        self.aggregate_ff = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Final classifier
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        x1: torch.LongTensor,
        x2: torch.LongTensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x1, x2: LongTensors of shape (batch, seq_len) with token IDs
            return_attention: if True, also return the attention matrices
        Returns:
            logits: (batch, num_classes)
            (optional) (alpha, beta): attention maps
        """
        # 1) Embed + project
        e1 = self.embedding(x1)              # (B, L1, E)
        e2 = self.embedding(x2)              # (B, L2, E)
        a = self.projection(e1)              # (B, L1, H)
        b = self.projection(e2)              # (B, L2, H)
        a = self.dropout(a)
        b = self.dropout(b)

        # 2) Attend step
        f_a = self.attend_ff(a)              # (B, L1, H)
        f_b = self.attend_ff(b)              # (B, L2, H)
        # similarity matrix
        e = torch.bmm(f_a, f_b.transpose(1, 2))  # (B, L1, L2)
        # Attention weights
        alpha = F.softmax(e, dim=2)          # (B, L1, L2)
        beta  = F.softmax(e, dim=1)          # (B, L1, L2)

        # Soft alignments
        align_b = torch.bmm(alpha, b)       # (B, L1, H)
        align_a = torch.bmm(beta.transpose(1, 2), a)  # (B, L2, H)

        # 3) Compare step
        comp_a = torch.cat([a, align_b], dim=2)  # (B, L1, 2H)
        comp_b = torch.cat([b, align_a], dim=2)  # (B, L2, 2H)
        v_a = self.compare_ff(comp_a)        # (B, L1, H)
        v_b = self.compare_ff(comp_b)        # (B, L2, H)

        # 4) Aggregate step (sum over time)
        v_a_sum = v_a.sum(dim=1)             # (B, H)
        v_b_sum = v_b.sum(dim=1)             # (B, H)
        v = torch.cat([v_a_sum, v_b_sum], dim=1)  # (B, 2H)

        # 5) Final MLP and classifier
        out = self.aggregate_ff(v)           # (B, H)
        logits = self.classifier(out)        # (B, num_classes)

        if return_attention:
            return logits, (alpha, beta)
        return logits
