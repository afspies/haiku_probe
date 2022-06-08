import haiku_template as sb
from pathlib import Path
import os.path as op

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds


sb.assign_free_gpus(threshold_vram_usage=3500)
sb.set_rng_seeds(seed=47)

data_path = Path(op.join(sb.__path__[0], 'data'))
# model = sb.modules.HaikuAutoInit()

import jax.numpy as jnp
with tf.device('CPU'):
    ds = tfds.load('mnist', split='train', shuffle_files=True, data_dir=data_path)
    # ds_iter = iter(ds)
    # print(next(ds_iter))
ds = ds.batch(32)
ds = ds.as_numpy_iterator()

import matplotlib.pyplot as plt

for batch in map(sb.transform_to_jax, ds):
    image, _ = batch[0].values()
    # image = ((image/255.0)-0.5)*2.0 # Normalize to [-1, 1]
    image = jnp.array(image/255.0, dtype=jnp.uint8)


# %%
