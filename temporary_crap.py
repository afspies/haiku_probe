import os.path as op
import numpy as np
import jax.numpy as jnp
import haiku as hk 
import haiku_probe as sb
import jax
# %env CUDA_VISIBLE_DEVICES=''
# %load_ext autoreload
# %autoreload 2

data_path = op.join(sb.__path__[0], 'data')