import torch
import torch.nn as nn
from einops import rearrange
from utils.utils import sinusoidal_positional_embedding
import constants as cst
import random
from models.diffusers.TRADES.Transformer import TransformerEncoder


class TRADES(nn.Module):
    def __init__(
        self,
        input_size,
        cond_seq_len,
        num_diffusionsteps,
        depth,
        num_heads,
        gen_sequence_size,
        cond_dropout_prob,
        is_augmented,
        dropout,
        cond_type,
        cond_method,
        use_news_features=False,
        news_feature_dim=2,
        lob_is_augmented=True,
        augment_dim=64
    ):
        super().__init__()
        self.cond_dropout_prob = cond_dropout_prob
        self.num_heads = num_heads
        self.use_news_features = use_news_features
        self.news_feature_dim = news_feature_dim if use_news_features else 0

        # Base input is the order features (without news)
        # News features will be passed via gated cross-attention, NOT concatenated
        if cond_method == 'concatenation' and cond_type == 'full' and is_augmented:
            # Add LOB dimension only (news not concatenated, handled via gated cross-attention)
            original_input_size = input_size  # Save original for output_size calculation
            # FIX: Use augmented LOB size when LOB is augmented, else use raw size
            lob_size = augment_dim if lob_is_augmented else (cst.N_LOB_LEVELS * cst.LEN_LEVEL)
            input_size = input_size + lob_size  # No news_feature_dim added here
            input_size = input_size + (input_size % 2)  # Pad to even dimension
            # FIX: output_size should match deaugmenter input (augmented orders + augmented LOB)
            if lob_is_augmented:
                # Both orders and LOB are augmented to augment_dim, so output is augment_dim * 2
                output_size = (augment_dim * 2) * gen_sequence_size
            else:
                output_size = original_input_size * gen_sequence_size
        elif cond_method == 'concatenation' and cond_type == 'full' and not is_augmented:
            output_size = input_size * gen_sequence_size
            # FIX: Use augmented LOB size when LOB is augmented, else use raw size
            lob_size = augment_dim if lob_is_augmented else (cst.N_LOB_LEVELS * cst.LEN_LEVEL)
            input_size = input_size + lob_size  # No news_feature_dim added here
        elif cond_method == "crossattention":
            output_size = input_size * gen_sequence_size
        else:
            # News will be passed via gated cross-attention, not concatenated
            output_size = input_size * gen_sequence_size

        self.input_size = input_size
        self.t_embedder = sinusoidal_positional_embedding(num_diffusionsteps, input_size) #TimestepEmbedder(input_size, input_size//4, num_diffusionsteps)
        self.seq_size = gen_sequence_size + cond_seq_len
        self.pos_embed = sinusoidal_positional_embedding(self.seq_size, input_size)
        self.is_augmented = is_augmented
        self.cond_method = cond_method
        self.cond_type = cond_type
        self.output_size = output_size
        self.gen_sequence_size = gen_sequence_size
        self.layers = TransformerEncoder(
            num_heads, input_size, depth, dropout, cond_type, cond_method,
            use_gated_cross_attn=use_news_features,
            news_feature_dim=self.news_feature_dim
        )
        self.fc_noise = nn.Linear(input_size*self.seq_size, output_size, device=cst.DEVICE)
        self.fc_var = nn.Linear(input_size*self.seq_size, output_size, device=cst.DEVICE)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x, cond_orders, t, cond_lob=None, cond_news=None):
        """
        Forward pass of TRADES.
        x: (N, K, F) tensor of time series
        t: (N,) tensor of diffusion timesteps
        cond_orders: (N, P, C) tensor of past history
        cond_lob: (N, P+1, L) tensor of LOB snapshots (optional)
        cond_news: (N, P, D) tensor of news features (optional) - passed via gated cross-attention
        """
        cond_orders = self.token_drop(cond_orders)
        full_input = torch.cat([cond_orders, x], dim=1)

        if self.gen_sequence_size > 1:
            cond_lob = torch.cat([cond_lob, torch.zeros((cond_lob.shape[0], self.gen_sequence_size-1, cond_lob.shape[2]), device=cond_lob.device)], dim=1)

        # Prepare news features for gated cross-attention (if enabled)
        cond_news_padded = None
        if self.use_news_features and cond_news is not None:
            # Pad news to match sequence length (add one more timestep and extend for gen_seq_size)
            # News shape: (N, cond_seq_size, news_dim) -> (N, cond_seq_size + gen_seq_size, news_dim)
            news_padding = torch.zeros((cond_news.shape[0], 1 + (self.gen_sequence_size - 1), cond_news.shape[2]),
                                      device=cond_news.device, dtype=cond_news.dtype)
            cond_news_padded = torch.cat([cond_news, news_padding], dim=1)

        # Concatenate LOB with orders (news is NOT concatenated, passed separately)
        if self.cond_method == 'concatenation' and self.cond_type == 'full':
            full_input = torch.cat([full_input, cond_lob], dim=-1)

        # Apply same augmentation as in __init__ to match pos_embed dimensions
        if self.is_augmented and full_input.shape[-1] % 2 == 1:
            # Pad with zeros to make dimension even (for sinusoidal embedding)
            padding = torch.zeros(full_input.shape[0], full_input.shape[1], 1,
                                 device=full_input.device, dtype=full_input.dtype)
            full_input = torch.cat([full_input, padding], dim=-1)

        full_input = full_input.add(self.pos_embed)
        diff_time_emb = self.t_embedder[t]
        full_input = full_input.add(diff_time_emb.view(diff_time_emb.shape[0], 1, diff_time_emb.shape[1]))
        full_input = self.layer_norm(full_input)

        # Pass through transformer with news features via gated cross-attention
        full_input = self.layers(full_input, news_features=cond_news_padded)

        full_input = rearrange(full_input, 'n l f -> n (l f)')
        noise = self.fc_noise(full_input)
        var = self.fc_var(full_input)
        noise = rearrange(noise, 'n (l d) -> n l d', l=self.gen_sequence_size, d=self.output_size//self.gen_sequence_size)
        var = rearrange(var, 'n (l d) -> n l d', l=self.gen_sequence_size, d=self.output_size//self.gen_sequence_size)
        return noise, var

    def token_drop(self, cond_orders):
        rand = random.random()
        if rand < self.cond_dropout_prob:
            # create a mask of zeros for the rows to drop
            mask = torch.zeros((cond_orders.shape), device=cond_orders.device)
            cond_orders = torch.einsum('bld, bld -> bld', cond_orders, mask)
            return cond_orders
        else:
            # no tokens are dropped
            return cond_orders 
        
 