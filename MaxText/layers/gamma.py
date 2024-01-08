import functools
import operator
from typing import Sequence, Union, Callable, Tuple, Any
from flax import linen as nn
import common_types
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from layers import models
from layers import normalizations
from layers import attentions
from layers import initializers
from layers import embeddings
from layers import linears
from layers.linears import _convert_to_activation_function

Embed = embeddings.Embed
RMSNorm = normalizations.RMSNorm
NdInitializer = initializers.NdInitializer
Attention = attentions.Attention
MlpBlock = linears.MlpBlock

LLaMARotaryEmbedding = embeddings.LLaMARotaryEmbedding

Config = common_types.Config
AxisNames = common_types.AxisNames
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn
DType = common_types.DType
Array = common_types.Array
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


nd_dense_init = initializers.nd_dense_init


# Decoder and Model definitions
class GammaDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: Config
  mesh: Mesh
  layer_idx: int = 0

  @nn.compact
  def __call__(self,
               inputs,
               decoder_segment_ids,
               decoder_positions,
               deterministic,
               model_mode,
              ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(
        inputs, ('activation_batch', 'activation_length', 'activation_embed'))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = RMSNorm(
        dtype=cfg.dtype,
        name='pre_self_attention_norm',
        kernel_axes=('embed',))(inputs)
    jax.debug.print("Mohit: shared_embed[0,0,0] after rmsnorm = {x}", x = lnx[0,:,0])

    lnx = nn.with_logical_constraint(
        lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    attention_layer = Attention(
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      attention_kernel=cfg.attention,
      mesh=mesh,
      dtype=cfg.dtype,
      dropout_rate=cfg.dropout_rate,
      name='self_attention',
      use_int8=cfg.int8_training)


    attention_lnx = attention_layer(
      lnx,
      lnx,
      decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=deterministic,
      model_mode=model_mode)
    # jax.debug.print("Mohit: shared_embed[0,0,0] after attention = {x}", x = attention_lnx[0,:,0])

    attention_lnx = nn.with_logical_constraint(
        attention_lnx,
        ('activation_batch', 'activation_length', 'activation_embed'))
    attention_lnx = inputs
    residual = attention_lnx
    jax.debug.print("Mohit: shared_embed[0,0,0] before attention rms = {x}", x = attention_lnx[0,:,0])
    attn_output = RMSNorm(
        dtype=cfg.dtype,
        name='pre_ffw_norm',
        kernel_axes=('embed',))(attention_lnx)
    jax.debug.print("Mohit: shared_embed[0,0,0] after attention rms = {x}", x = attn_output[0,:,0])
    # MLP block.
    mlp_lnx = MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
        config=cfg,
    )(attn_output, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(
        mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
    )
    jax.debug.print("Mohit: shared_embed[0,0,0] after mlp = {x}", x = mlp_lnx[0,:,0])
    next_layer_addition = mlp_lnx + residual
    jax.debug.print("Mohit: shared_embed[0,0,0] after addition = {x}", x = next_layer_addition[0,:,0])
    next_layer_addition_dropped_out = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,)
    )(next_layer_addition, deterministic=deterministic)

    # layer_output = next_layer_addition_dropped_out + inputs
    layer_output = next_layer_addition_dropped_out
    layer_output = nn.with_logical_constraint(
        layer_output,
        ('activation_batch', 'activation_length', 'activation_embed'),
    )

    if cfg.record_internal_nn_metrics:
      self.sow('intermediates', 'activation_mean', jnp.mean(layer_output))
      self.sow('intermediates', 'activation_stdev', jnp.std(layer_output))
      self.sow(
          'intermediates',
          'activation_fraction_zero',
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output