import argparse
import copy
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax.sharding import Mesh
from flax.training import train_state
from jax import random
import jax.numpy as jnp

from typing import Any
import sys
import max_logging


import orbax

import maxtext_utils
import max_utils
from layers import models

import pyconfig
import checkpointing

Transformer = models.Transformer
Params = dict[str, Any]


def assess_train_chkpt(config):
  init_rng, nextrng = random.split(random.PRNGKey(config.init_weights_seed), 2)

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
    config.checkpoint_dir,
    config.enable_checkpointing,
    config.async_checkpointing,
    config.save_period,
  )

  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  model = Transformer(config, mesh)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
  tx = maxtext_utils.get_optimizer(config, learning_rate_schedule)
  state, _ = max_utils.setup_training_state(model, tx, config, init_rng, mesh, checkpoint_manager)
  print(state.params.keys())

def nest_params(params: Params) -> Params:
  """Nests params as a dict of dicts rather than a flat dict."""
  nested_params = {}
  for path, param in params.items():
    *path, leaf = path.split('/')
    subdict = nested_params
    for key in path:
      subdict = subdict.setdefault(key, {})
    subdict[leaf] = param
  return nested_params

def main(raw_args=None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_model_path', type=str, required=True)
  parser.add_argument('--maxtext_model_path', type=str, required=True)
  args = parser.parse_args(raw_args)

  print("Loading checkpoint")
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  params = checkpointer.restore(args.base_model_path) #'gs://mohitkhatwani-maxtext-chkpts/gamma/7b_alpha_pt_orbax/'
  params = nest_params(params)
  num_layers = (
      max([
          int(k.split('_')[1])
          for k in params['transformer'].keys()
          if 'layer_' in k
      ])
      + 1
  )
  hidden_dim, embed_dim = (
        params['transformer']['layer_0']['mlp']['linear'].shape
    )
  num_heads, head_dim, _ = (
      params['transformer']['layer_0']['attn']['attn_vec_einsum']['w'].shape
  )
  print("Model configurations from checkpoint")
  print(f"num_layers: {num_layers}")
  print(f"hidden_dim: {hidden_dim}")
  print(f"embed_dim: {embed_dim}")
  print(f"num_heads: {num_heads}")
  print(f"head_dim: {head_dim}")
  
  jax_weights = {
    'decoder': {
        'decoder_norm': {
          'scale': params['transformer']['final_norm']['scale'] + 1
        },     
      },
      'token_embedder':{
        'embedding': params['transformer']['embedder']['input_embedding'] * jnp.sqrt(embed_dim)
      }

  }
  self_attention = dict({
      'query': {
          'kernel' : []
      },
      'key': {
          'kernel' : []
      },
      'value': {
          'kernel' : []
      },
      'out': {
          'kernel' : []
      },
  })

  layer_weight = dict({
    'mlp': {
      'wi_0': {
          'kernel' : []
          },
      'wi_1': {
          'kernel' : []
          },
      'wo': {
          'kernel' : []
          },
    },
    'pre_self_attention_norm': {
        'scale': []
    },
    'pre_ffw_norm': {
      'scale': []
    },
  })
  
  for layer_idx in range(num_layers):
    in_layer_name = 'layer_' + str(layer_idx)
    # attention block
    self_attention['query']['kernel'] = params['transformer'][in_layer_name]['attn']['qkv_einsum']['w'][0].transpose((1, 0, 2)) * jnp.sqrt(embed_dim) ** -0.5 
    self_attention['key']['kernel'] = params['transformer'][in_layer_name]['attn']['qkv_einsum']['w'][1].transpose((1, 0, 2))
    self_attention['value']['kernel'] = params['transformer'][in_layer_name]['attn']['qkv_einsum']['w'][2].transpose((1, 0, 2))
    self_attention['out']['kernel'] = params['transformer'][in_layer_name]['attn']['attn_vec_einsum']['w']
    # mlp
    if 'w' not in params['transformer'][in_layer_name]['mlp']['gating_einsum']: # the weights are remapped
      layer_weight['mlp']['wi_0']['kernel'] = params['transformer'][in_layer_name]['mlp']['gating_einsum'][0]
      layer_weight['mlp']['wi_1']['kernel'] = params['transformer'][in_layer_name]['mlp']['gating_einsum'][1]
      layer_weight['mlp']['wo']['kernel'] = params['transformer'][in_layer_name]['mlp']['linear']
    else:
      layer_weight['mlp']['wi_0']['kernel'] = params['transformer'][in_layer_name]['mlp']['gating_einsum']['w'][0]
      layer_weight['mlp']['wi_1']['kernel'] = params['transformer'][in_layer_name]['mlp']['gating_einsum']['w'][1]
      layer_weight['mlp']['wo']['kernel'] = params['transformer'][in_layer_name]['mlp']['linear']['w']
    layer_weight['pre_self_attention_norm']['scale'] = params['transformer'][in_layer_name]['pre_attention_norm']['scale'] + 1
    layer_weight['pre_ffw_norm']['scale'] = params['transformer'][in_layer_name]['pre_ffw_norm']['scale'] + 1
    layer_weight['self_attention'] = copy.deepcopy(self_attention)
    jax_weights['decoder']['layers_' + str(layer_idx)] = copy.deepcopy(layer_weight)

  enable_checkpointing=True
  async_checkpointing=False
  save_interval_steps=1


  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      args.maxtext_model_path,
      enable_checkpointing,
      async_checkpointing,
      save_interval_steps
  )

  state_new = train_state.TrainState(
    step=0,
    apply_fn=None,
    params=jax_weights,
    tx=None, # type: ignore
    opt_state={}
  )

  if checkpoint_manager is not None:
    if checkpoint_manager.save(0, state_new):
      max_logging.log(f"saved a checkpoint at step 0")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()

if __name__ == "__main__":
  main()