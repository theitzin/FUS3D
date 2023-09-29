# motified from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py

import math
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce, Rearrange

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Sequential(
            nn.Linear(query_dim, inner_dim, bias=False),
            Rearrange('b n (h d) -> (b h) n d', h=self.heads)
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=False),
            Rearrange('b n (h d) -> (b h) n d', h=self.heads)
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=False),
            Rearrange('b n (h d) -> (b h) n d', h=self.heads)
        )

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        context = default(context, x)
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            if len(mask.shape) == 2:
                mask = rearrange(mask, 'b j -> b 1 j')
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out), sim

# main class

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        self_per_cross_attn = 1,
        final_classifier_head = True
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_channels, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_channels)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        latent_mask = None,
        return_embeddings = False,
        latents = None
    ):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # concat to channels of data and flatten axis
        data = rearrange(data, 'b ... d -> b (...) d')

        if latents is None:
            x = repeat(self.latents, 'n d -> b n d', b = b)
        else:
            x = latents

        # layers
        for cross_attn, cross_ff, self_attns in self.layers:
            r1, sim = cross_attn(x, context=data, mask=mask)
            x = r1 + x
            r2 = cross_ff(x)
            x = r2 + x

            for self_attn, self_ff in self_attns:
                r3, _ = self_attn(x, mask=latent_mask)
                x = r3 + x
                r4 = self_ff(x)
                x = r4 + x

        # allow for fetching embeddings
        if return_embeddings:
            return x

        # to logits
        return self.to_logits(x)


class DetrSinePositionEmbedding(nn.Module):
    def __init__(self, embedding_dim=64, temperature=5):
        super().__init__()
        assert embedding_dim % 2 == 0
        self.embedding_dim = embedding_dim // 2
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, positions):
        y_embed = positions[..., 1] * self.scale
        x_embed = positions[..., 0] * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=positions.device)
        dim_t = self.temperature ** (2 * dim_t / self.embedding_dim)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.cat((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=2)
        pos_y = torch.cat((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos