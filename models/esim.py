import torch
import torch.nn as nn
import torch.nn.functional as F


class ESIM(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super().__init__()

        # Embedding (frozen)
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=True,
            padding_idx=padding_idx,
        )
        embed_dim = embedding_matrix.size(1)

        # BiLSTM encoders
        self.encoder1 = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.encoder2 = nn.LSTM(
            input_size=hidden_size * 8,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 8, 1024),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(
        self,
        x1: torch.LongTensor,
        len1: torch.LongTensor,
        x2: torch.LongTensor,
        len2: torch.LongTensor,
        return_attention: bool = False,
    ):
        # 1) Embed
        v1 = self.embedding(x1)
        v2 = self.embedding(x2)

        # 2) First BiLSTM (includes padding)
        o1, _ = self.encoder1(v1)
        o2, _ = self.encoder1(v2)

        # 3) Soft attention
        e  = torch.bmm(o1, o2.transpose(1, 2))
        a1 = F.softmax(e, dim=2)
        a2 = F.softmax(e, dim=1)

        # 4) Attend
        t1 = torch.bmm(a1, o2)
        t2 = torch.bmm(a2.transpose(1, 2), o1)

        # 5) Inference composition inputs
        m1 = torch.cat([o1, t1, o1 - t1, o1 * t1], dim=2)
        m2 = torch.cat([o2, t2, o2 - t2, o2 * t2], dim=2)

        # 6) Second BiLSTM (includes padding)
        o1_hat, _ = self.encoder2(m1)
        o2_hat, _ = self.encoder2(m2)

        # 7) Masks for pooling
        mask1 = (torch.arange(x1.size(1), device=len1.device)
                 .unsqueeze(0) < len1.unsqueeze(1))
        mask2 = (torch.arange(x2.size(1), device=len2.device)
                 .unsqueeze(0) < len2.unsqueeze(1))

        # 8) Masked mean pooling
        avg1 = (o1_hat * mask1.unsqueeze(2)).sum(1) / len1.unsqueeze(1)
        avg2 = (o2_hat * mask2.unsqueeze(2)).sum(1) / len2.unsqueeze(1)

        # 9) Masked max pooling
        o1_hat = o1_hat.masked_fill(~mask1.unsqueeze(2), float('-inf'))
        o2_hat = o2_hat.masked_fill(~mask2.unsqueeze(2), float('-inf'))
        max1, _ = o1_hat.max(1)
        max2, _ = o2_hat.max(1)

        # 10) Concat and classify
        v = torch.cat([avg1, max1, avg2, max2], dim=1)
        logits = self.classifier(v)

        if return_attention:
            return logits, (a1, a2)
        return logits