import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from flax import linen as nn
from flax.linen.initializers import zeros
from typing import Callable, Sequence

import chex
import jax
import jax.numpy as jnp
import math
import numpy as np

from src.constants import *


class MLPModule(nn.Module):
    """Multilayer Perceptron."""

    # The number of hidden units in each hidden layer.
    layers: Sequence[int]
    activation: Callable
    output_activation: Callable
    use_batch_norm: bool
    use_layer_norm: bool
    use_bias: bool
    flatten: bool = False

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        idx = -1
        if self.flatten:
            x = x.reshape((len(x), -1))
        if self.use_batch_norm:
            x = nn.BatchNorm()(x, eval)
        for idx, layer in enumerate(self.layers[:-1]):
            x = nn.Dense(layer)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation(x)
            if self.use_batch_norm:
                x = nn.BatchNorm()(x, eval)
            self.sow("intermediates", f"activation_{idx}", x)
        x = self.output_activation(nn.Dense(self.layers[-1], use_bias=self.use_bias)(x))
        self.sow("intermediates", f"activation_{idx}", x)
        return x


class CNNModule(nn.Module):
    """Convolutional layer."""

    features: Sequence[int]
    kernel_sizes: Sequence[Sequence[int]]
    activation: Callable
    use_batch_norm: bool

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        for idx, (feature, kernel_size) in enumerate(
            zip(self.features, self.kernel_sizes)
        ):
            x = self.activation(nn.Conv(feature, kernel_size)(x))
            if self.use_batch_norm:
                x = nn.BatchNorm(
                    use_scale=True,
                )(x, eval)
            self.sow("intermediates", f"activation_{idx}", x)
        return x


class SelfAttentionModule(nn.Module):
    """Self-Attention layer."""

    num_heads: int
    qkv_features: int = None

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, mask=None, **kwargs) -> chex.Array:
        in_dim = x.shape[-1]

        if self.qkv_features is not None:
            qkv_hiddens = self.qkv_features
        else:
            qkv_hiddens = in_dim

        q = nn.Dense(qkv_hiddens)(x)
        k = nn.Dense(qkv_hiddens)(x)
        v = nn.Dense(qkv_hiddens)(x)

        batch, q_time, _ = q.shape
        _, kv_time, _ = k.shape
        head_dim = qkv_hiddens // self.num_heads
        q = jnp.reshape(q, [batch, q_time, self.num_heads, head_dim])
        k = jnp.reshape(k, [batch, kv_time, self.num_heads, head_dim])
        v = jnp.reshape(v, [batch, kv_time, self.num_heads, head_dim])

        # attend
        hiddens = self.num_heads * head_dim
        scale = 1.0 / math.sqrt(head_dim)
        attention = jnp.einsum("bthd,bThd->bhtT", q, k)
        attention *= scale
        if mask is not None:
            attention = attention * mask - 1e10 * (1 - mask)
        normalized = jax.nn.softmax(attention)
        summed = jnp.einsum("bhtT,bThd->bthd", normalized, v)
        out = jnp.reshape(summed, [batch, q_time, hiddens])

        return nn.Dense(qkv_hiddens)(out)


class GPTBlock(nn.Module):
    """GPT Block."""

    # : The number of attention heads
    num_heads: int

    # : The embedding dimensionality
    embed_dim: int

    # : Widening dimension
    widening_factor: int

    # : Use causal mask
    causal_mask: float = 1.0

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        mask = nn.make_causal_mask(x[..., 0]) * self.causal_mask
        x = x + SelfAttentionModule(self.num_heads, self.embed_dim)(
            nn.LayerNorm()(x), eval, mask=mask
        )
        normed_x = nn.gelu(
            nn.Dense(self.embed_dim * self.widening_factor)(nn.LayerNorm()(x))
        )
        x = x + nn.Dense(self.embed_dim)(normed_x)
        return x


class GPTModule(nn.Module):
    """GPT."""

    # : The number of GPT Blocks
    num_blocks: int

    # : The number of attention heads
    num_heads: int

    # : The embedding dimensionality
    embed_dim: int

    # : Widening dimension
    widening_factor: int

    # : Use causal mask
    causal_mask: float = 1.0

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        for idx, _ in enumerate(range(self.num_blocks)):
            x = GPTBlock(
                self.num_heads, self.embed_dim, self.widening_factor, self.causal_mask
            )(x, eval)
        x = nn.LayerNorm()(x)
        return x


class PatchEmbedding(nn.Module):
    patch_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        embeddings = nn.Conv(
            self.embed_dim,
            kernel_size=[self.patch_size, self.patch_size],
            strides=[self.patch_size, self.patch_size],
        )(x)

        N = embeddings.shape[:-3]
        (H, W, D) = embeddings.shape[-3:]

        return embeddings.reshape((*N, H * W, D))


class RandomEmbedding(nn.Module):
    embed_dim: int
    fixed_value: float = None

    @nn.compact
    def __call__(self, **kwargs) -> jnp.ndarray:
        random_embedding = self.param(
            "random_embedding",
            (
                nn.initializers.normal()
                if self.fixed_value is None
                else nn.initializers.constant(self.fixed_value)
            ),
            (self.embed_dim,),
        )
        return random_embedding


class ResNetV1Block(nn.Module):
    """
    ResNet V1 Block.
    Reference: https://github.com/google-deepmind/emergent_in_context_learning/blob/eba75a4208b8927cc1e981384a2cc7e014677095/modules/resnet.py
    """

    features: int
    stride: Sequence[int]
    use_projection: bool
    use_bottleneck: bool
    use_batch_norm: bool

    def setup(self):
        assert (
            not self.use_bottleneck or self.features >= 4 and self.features % 4 == 0
        ), "must have at least 4n kernels {} when using bottleneck".format(
            self.features
        )
        if self.use_projection:
            self.projection = nn.Conv(
                self.features,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
                padding=CONST_SAME_PADDING,
            )
            if self.use_batch_norm:
                self.projection_batchnorm = nn.BatchNorm(
                    momentum=0.9,
                    epsilon=1e-5,
                    use_bias=True,
                    use_scale=True,
                    use_fast_variance=False,
                )

        conv_features = self.features
        conv_0_kernel = (3, 3)
        conv_0_stride = self.stride
        conv_1_stride = 1
        if self.use_bottleneck:
            conv_features = self.features // 4
            conv_0_kernel = (1, 1)
            conv_0_stride = 1
            conv_1_stride = self.stride

        self.conv_0 = nn.Conv(
            conv_features,
            kernel_size=conv_0_kernel,
            strides=conv_0_stride,
            use_bias=False,
            padding=CONST_SAME_PADDING,
        )

        if self.use_batch_norm:
            self.batch_norm_0 = nn.BatchNorm(
                momentum=0.9,
                epsilon=1e-5,
                use_bias=True,
                use_scale=True,
                use_fast_variance=False,
            )

        self.conv_1 = nn.Conv(
            conv_features,
            kernel_size=(3, 3),
            strides=conv_1_stride,
            use_bias=False,
            padding=CONST_SAME_PADDING,
        )

        if self.use_batch_norm:
            self.batch_norm_1 = nn.BatchNorm(
                momentum=0.9,
                epsilon=1e-5,
                use_bias=True,
                use_scale=True,
                use_fast_variance=False,
            )

        if self.use_batch_norm:
            layers = [
                (self.conv_0, self.batch_norm_0),
                (self.conv_1, self.batch_norm_1),
            ]
        else:
            layers = [self.conv_0, self.conv_1]

        if self.use_bottleneck:
            self.conv_2 = nn.Conv(
                self.features,
                kernel_size=(1, 1),
                strides=1,
                use_bias=False,
                padding=CONST_SAME_PADDING,
            )
            if self.use_batch_norm:
                self.batch_norm_2 = nn.BatchNorm(
                    momentum=0.9,
                    epsilon=1e-5,
                    use_bias=True,
                    use_scale=True,
                    scale_init=zeros,
                    use_fast_variance=False,
                )
                layers.append((self.conv_2, self.batch_norm_2))
            else:
                layers.append(self.conv_2)
        self.layers = layers

    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        out = shortcut = x

        if self.use_projection:
            shortcut = self.projection(shortcut)
            if self.use_batch_norm:
                shortcut = self.projection_batchnorm(shortcut, eval)

        idx = -1

        if self.use_batch_norm:
            for idx, (conv_i, batch_norm_i) in enumerate(self.layers[:-1]):
                out = conv_i(out)
                out = batch_norm_i(out, eval)
                out = jax.nn.relu(out)
                self.sow("intermediates", f"activation_{idx}", out)
            out = self.layers[-1][0](out)
            out = self.layers[-1][1](out, eval)
            self.sow("intermediates", f"activation_{idx}", out)
        else:
            for idx, conv_i in enumerate(self.layers[:-1]):
                out = conv_i(out)
                out = jax.nn.relu(out)
                self.sow("intermediates", f"activation_{idx}", out)
            out = self.layers[-1](out)
            self.sow("intermediates", f"activation_{idx}", out)

        out = jax.nn.relu(out + shortcut)
        self.sow("intermediates", f"activation_output", out)
        return out


class ResNetV1BlockGroup(nn.Module):
    num_blocks: int
    features: int
    stride: Sequence[int]
    use_projection: bool
    use_bottleneck: bool
    use_batch_norm: bool

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        for block_i in range(self.num_blocks):
            x = ResNetV1Block(
                self.features,
                1 if block_i else self.stride,
                block_i == 0 and self.use_projection,
                self.use_bottleneck,
                self.use_batch_norm,
            )(x, eval)
        return x


class ResNetV1Module(nn.Module):
    blocks_per_group: Sequence[int]
    features: Sequence[int]
    stride: Sequence[Sequence[int]]
    use_projection: Sequence[bool]
    use_bottleneck: bool
    use_batch_norm: bool

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=2,
            use_bias=False,
            padding=CONST_SAME_PADDING,
        )(x)

        if self.use_batch_norm:
            x = nn.BatchNorm(
                momentum=0.9,
                epsilon=1e-5,
                use_bias=True,
                use_scale=True,
                use_fast_variance=False,
            )(x, eval)
        x = jax.nn.relu(x)
        x = nn.max_pool(
            x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding=CONST_SAME_PADDING,
        )
        self.sow("intermediates", f"activation_projection", x)

        for (
            curr_blocks,
            curr_features,
            curr_stride,
            curr_projection,
        ) in zip(
            self.blocks_per_group, self.features, self.stride, self.use_projection
        ):
            x = ResNetV1BlockGroup(
                curr_blocks,
                curr_features,
                curr_stride,
                curr_projection,
                self.use_bottleneck,
                self.use_batch_norm,
            )(x, eval)
        return jnp.mean(x, axis=(-3, -2))


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self, **kwargs) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda _: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)

    def update_batch_stats(self, params, batch_stats):
        return params


class PositionalEncoding(nn.Module):
    """
    Default positional encoding used in Transformers. More correct implementation following Chan et al.?
    Reference: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
    """

    embed_dim: int
    max_len: int
    period: float = 30.0

    def setup(self):
        pe = np.zeros((self.max_len, self.embed_dim))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.log(self.period) * (-np.arange(0, self.embed_dim, 2) / self.embed_dim)
        )
        half_dim = self.embed_dim // 2
        pe[:, :half_dim] = np.sin(position * div_term)
        pe[:, half_dim:] = np.cos(position * div_term)
        self.pe = pe[None]

    def __call__(self, x: chex.Array, **kwargs):
        x = x + self.pe[:, : x.shape[1]]
        return x
