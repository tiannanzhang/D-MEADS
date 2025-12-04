from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self,
                num_heads: int,
                d_model: int,
                num_layers: int,
                dropout: float,
                cond_type: str,
                cond_method: str,
                use_gated_cross_attn: bool = False,
                news_feature_dim: int = 0
                ):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_gated_cross_attn = use_gated_cross_attn

        if cond_type == 'full' and cond_method == 'crossattention':
            self.layers = nn.ModuleList(
                [block(d_model, num_heads, dropout) for _ in range(num_layers//2) for block in (TransformerBlockSelfAtt, TransformerBlockCrossAtt)]
            )
        else:
            # Create self-attention blocks with optional gated cross-attention
            self.layers = nn.ModuleList([
                TransformerBlockSelfAtt(d_model, num_heads, dropout, use_gated_cross_attn, news_feature_dim)
                for _ in range(num_layers)
            ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, cond=None, news_features=None):
        for layer in self.layers:
            if self.use_gated_cross_attn:
                x = layer(x, mask, cond, news_features)
            else:
                x = layer(x, mask, cond)
        return x

class TransformerBlockCrossAtt(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(TransformerBlockCrossAtt, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.PReLU(init=0.01),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.num_heads = num_heads
        self.d_model = d_model
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, cond=None):
        # x.shape = B, L, D
        # mask.shape = L, L

        # firstly we compute q, k, v and divide them into heads
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)
        q, k, v = map(lambda t: rearrange(t, 'b l (h j) -> b h l j', h=self.num_heads), (q, k, v))
        # Scale the query (q) by the square root of the dimensionality of the model (d_model)
        q = torch.mul(q, 1/(self.d_model ** 0.5))
        # Compute the dot product of the query (q) and key (k) to get the raw attention scores (e)
        e = torch.einsum('b h l j, b h k i -> b h l k', q, k)
        # Apply the mask to the raw attention scores (e)
        if mask is not None:
            e = torch.add(e, mask)
        # Apply the softmax function to the masked attention scores to get the attention weights (att)
        att = torch.nan_to_num(F.softmax(e, dim=-1))
        # Multiply the attention weights (att) with the value (v) to get the output of the attention mechanism
        out_att = torch.einsum('b h l l, b h l j -> b h l j', att, v)
        out_att = rearrange(out_att, 'b h l j -> b l (h j)')
        # Pass the output through the output linear layer
        out_att = self.to_out(out_att)
        # Apply the residual connection and layer normalization
        out_att = self.layer_norm1(out_att + x)
        # Pass the output through the MLP
        out = self.mlp(out_att)

        return self.layer_norm2(out + out_att)

class GatedCrossAttention(nn.Module):
    """
    Gated cross-attention module for conditioning on news features.
    The gate is initialized to 0 to preserve pretrained model behavior at start.
    """
    def __init__(self, d_model, num_heads, news_feature_dim, dropout):
        super(GatedCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Project news features to d_model dimension
        self.news_projection = nn.Linear(news_feature_dim, d_model)

        # Cross-attention layers
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

        # Gate parameter - initialized to 0 for pretrained model preservation
        self.gate = nn.Parameter(torch.zeros(1))

        self.layer_norm = nn.LayerNorm(d_model)
        # NOTE: No dropout after to_out - matches original TransformerBlockCrossAtt architecture

    def forward(self, x, news_features):
        """
        x: (B, L, D) - main features (query)
        news_features: (B, L, news_dim) - news features (context for cross-attention)
        """
        # DEBUG: Check input
        if torch.isnan(x).any():
            print(f"[GatedCrossAttn] Input x has NaN!")
        if torch.isnan(news_features).any():
            nan_pct = 100 * torch.isnan(news_features).sum().item() / news_features.numel()
            print(f"[GatedCrossAttn] Input news_features has {nan_pct:.1f}% NaN")

        # Handle NaN values in news features (replace with 0)
        # NaN indicates no news data at that timestep - neutral signal
        news_features = torch.nan_to_num(news_features, nan=0.0)

        # Project news features to d_model
        news_proj = self.news_projection(news_features)  # (B, L, D)
        if torch.isnan(news_proj).any() or torch.isinf(news_proj).any():
            print(f"[GatedCrossAttn] news_proj has NaN/Inf! min={news_proj.min().item()}, max={news_proj.max().item()}")

        # Compute cross-attention: x is query, news_proj is key and value
        q = self.to_q(x)
        k = self.to_k(news_proj)
        v = self.to_v(news_proj)

        if torch.isnan(q).any():
            print(f"[GatedCrossAttn] q has NaN!")
        if torch.isnan(k).any():
            print(f"[GatedCrossAttn] k has NaN!")
        if torch.isnan(v).any():
            print(f"[GatedCrossAttn] v has NaN!")

        # Split into heads
        q, k, v = map(lambda t: rearrange(t, 'b l (h j) -> b h l j', h=self.num_heads), (q, k, v))

        # Scaled dot-product attention
        q = torch.mul(q, 1/(self.d_model ** 0.5))
        e = torch.einsum('b h l j, b h k i -> b h l k', q, k)

        if torch.isnan(e).any() or torch.isinf(e).any():
            print(f"[GatedCrossAttn] Attention logits e has NaN/Inf!")

        att = torch.nan_to_num(F.softmax(e, dim=-1))

        # Apply attention to values
        out_att = torch.einsum('b h l k, b h k j -> b h l j', att, v)
        out_att = rearrange(out_att, 'b h l j -> b l (h j)')
        out_att = self.to_out(out_att)

        if torch.isnan(out_att).any():
            print(f"[GatedCrossAttn] out_att has NaN before gate!")

        # Apply gate and residual connection
        gated_out = self.gate * out_att
        if torch.isnan(gated_out).any():
            print(f"[GatedCrossAttn] gated_out has NaN after gate multiplication!")

        out = self.layer_norm(x + gated_out)

        if torch.isnan(out).any():
            print(f"[GatedCrossAttn] Final output has NaN!")

        return out

class TransformerBlockSelfAtt(nn.Module):
    def __init__(self, d_model, num_heads, dropout, use_gated_cross_attn=False, news_feature_dim=0):
        super(TransformerBlockSelfAtt, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.PReLU(init=0.01),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.num_heads = num_heads
        self.d_model = d_model
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Add gated cross-attention if requested
        self.use_gated_cross_attn = use_gated_cross_attn
        if use_gated_cross_attn:
            self.gated_cross_attn = GatedCrossAttention(d_model, num_heads, news_feature_dim, dropout)

    def forward(self, x, mask=None, cond=None, news_features=None):
        # x.shape = B, L, D
        # mask.shape = L, L

        # firstly we compute q, k, v and divide them into heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b l (h j) -> b h l j', h=self.num_heads), (q, k, v))
        # Scale the query (q) by the square root of the dimensionality of the model (d_model)
        q = torch.mul(q, 1/(self.d_model ** 0.5))
        # Compute the dot product of the query (q) and key (k) to get the raw attention scores (e)
        e = torch.einsum('b h l j, b h k i -> b h l k', q, k)
        # Apply the mask to the raw attention scores (e)
        if mask is not None:
            e = torch.add(e, mask)
        # Apply the softmax function to the masked attention scores to get the attention weights (att)
        att = torch.nan_to_num(F.softmax(e, dim=-1))
        # Multiply the attention weights (att) with the value (v) to get the output of the attention mechanism
        out_att = torch.einsum('b h l l, b h l j -> b h l j', att, v)
        out_att = rearrange(out_att, 'b h l j -> b l (h j)')
        # Pass the output through the output linear layer
        out_att = self.to_out(out_att)
        # Apply the residual connection and layer normalization
        out_att = self.layer_norm1(out_att + x)

        # Apply gated cross-attention if enabled and news features are provided
        if self.use_gated_cross_attn and news_features is not None:
            out_att = self.gated_cross_attn(out_att, news_features)

        # Pass the output through the MLP
        out = self.mlp(out_att)

        return self.layer_norm2(out + out_att)