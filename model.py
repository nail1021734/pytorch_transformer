import torch
import math
import numpy as np


class FeedForward(torch.nn.Module):
    def __init__(self, d_hid, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w1 = torch.nn.Linear(
            in_features=d_hid,
            out_features=d_ff
        )
        self.w2 = torch.nn.Linear(
            in_features=d_ff,
            out_features=d_hid
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(self.dropout(torch.nn.functional.relu(self.w1(x)))))


class AddNorm(torch.nn.Module):
    def __init__(self, feature_num, eps=1e-6):
        super().__init__()
        self.scale_a = torch.nn.Parameter(torch.ones(feature_num))
        self.scale_b = torch.nn.Parameter(torch.zeros(feature_num))
        self.eps = eps

    def forward(self, x, sub):
        x = x + sub
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.scale_a * (x - mean) / (std + self.eps) + self.scale_b


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.dropout = torch.nn.Dropout(p=dropout)
        self.v_linear = torch.nn.Linear(
            in_features=d_model,
            out_features=d_model
        )
        self.k_linear = torch.nn.Linear(
            in_features=d_model,
            out_features=d_model
        )
        self.q_linear = torch.nn.Linear(
            in_features=d_model,
            out_features=d_model
        )
        self.out_linear = torch.nn.Linear(
            in_features=d_model,
            out_features=d_model
        )

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # mask (B,1,L)
            mask = mask.unsqueeze(1)
        batch_num = query.size(0)

        query = self.q_linear(query).view(batch_num, -1, self.h, self.d_k)
        key = self.k_linear(key).view(batch_num, -1, self.h, self.d_k)
        value = self.v_linear(value).view(batch_num, -1, self.h, self.d_k)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).reshape(batch_num, -1, self.d_model)
        return self.out_linear(x)

    def attention(self, query, key, value, mask=None, dropout=None):
        similarity = torch.matmul(query, key.transpose(-2, -1))
        # (B,h,L,L)
        similarity = similarity / math.sqrt(self.d_k)

        if mask is not None:
            similarity = similarity.masked_fill(mask == 0, -1e9)
        attn = torch.nn.functional.softmax(similarity, dim=-1)

        if dropout is not None:
            attn = dropout(attn)
        attn = torch.matmul(attn, value)

        return attn


class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + \
            torch.autograd.Variable(
                self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.MHA = MultiHeadAttention(d_model, h)
        self.addnormlayer1 = AddNorm(d_model)
        self.FF = FeedForward(d_model)
        self.addnormlayer2 = AddNorm(d_model)

    def forward(self, x, mask=None):
        x = self.addnormlayer1(x, self.MHA(x, x, x, mask))
        return self.addnormlayer2(x, self.FF(x))


class Encoder(torch.nn.Module):
    def __init__(self, d_model, h, N):
        super().__init__()
        self.Encodelayers = torch.nn.ModuleList(
            [EncoderLayer(d_model, h) for _ in range(N)])

    def forward(self, x, mask):
        for layer in self.Encodelayers:
            x = layer(x, mask)

        return x


class DecodeLayer(torch.nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.MHA1 = MultiHeadAttention(d_model, h)
        self.addnormlayer1 = AddNorm(d_model)
        self.MHA2 = MultiHeadAttention(d_model, h)
        self.addnormlayer2 = AddNorm(d_model)
        self.FF = FeedForward(d_model)
        self.addnormlayer3 = AddNorm(d_model)

    def forward(self, x, en_output, src_mask, tgt_mask):
        x = self.addnormlayer1(x, self.MHA1(x, x, x, tgt_mask))
        x = self.addnormlayer2(x, self.MHA2(x, en_output, en_output, src_mask))
        x = self.addnormlayer3(x, self.FF(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, d_model, h, N):
        super().__init__()
        self.Decoderlayers = torch.nn.ModuleList(
            [DecodeLayer(d_model, h) for _ in range(N)])

    def forward(self, x, en_output, src_mask, tgt_mask):
        for layer in self.Decoderlayers:
            x = layer(x, en_output, src_mask, tgt_mask)
        return x


class Mask:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def src_mask(self, x):
        # (B, 1, L)
        return (x != self.pad_idx).unsqueeze(-2)

    def tgt_mask(self, x):
        tgt_mask = self.src_mask(x)
        triu_mask = torch.from_numpy(
            np.triu(
                np.ones((1, x.size(-1), x.size(-1)), dtype=np.uint8),
                k=1
            )
        ).to(x.device)
        triu_mask = (triu_mask == 0)
        return tgt_mask & triu_mask


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        h,
        encoder_N,
        decoder_N,
        vocab_size,
        pad_token_id,
        dropout
    ):
        super().__init__()
        self.mask = Mask(pad_token_id)
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id
        )
        self.PE = PositionalEncoding(d_model, dropout)
        self.Encodelayers = Encoder(d_model, h=h, N=encoder_N)
        self.DecodeLayers = Decoder(d_model, h=h, N=decoder_N)
        self.linear = torch.nn.Linear(
            in_features=d_model,
            out_features=d_model
        )

    def forward(self, x, y):
        src_mask = self.mask.src_mask(x)
        tgt_mask = self.mask.tgt_mask(y)
        x = self.embedding_layer(x)
        x = self.PE(x)
        x = self.Encodelayers(x, src_mask)
        y = self.embedding_layer(y)
        y = self.PE(y)
        y = self.DecodeLayers(y, x, src_mask, tgt_mask)
        y = self.linear(y)
        y = y @ self.embedding_layer.weight.transpose(0, 1)

        return y

    def predict(self, x: torch.Tensor, y: torch.Tensor):
        return torch.nn.functional.softmax(self(x, y), dim=-1)

# if __name__ == "__main__":
#     # (B,L)
#     x = torch.tensor([[1, 2, 3], [4, 5, 6]])
#     y = torch.tensor([[3, 3, 9, 4, 5, 6], [4, 4, 8, 4, 4 ,4]])
#     model = Transformer(d_model=512, h=8, encoder_N=1,
#                         decoder_N=1, vocab_size=10, pad_token_id=9, dropout=0.1)
#     out = model.predict(x, y)
#     print(out.shape)
