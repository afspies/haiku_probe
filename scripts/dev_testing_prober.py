import haiku as hk
import jax
import jax.numpy as jnp
from haiku_probe.modules.networks import TestingNetwork, HaikuAutoInit
from haiku_probe.modules.prober import create_probe

cfg = {'linear':16, 'seed':42}
class SimpleLinear(TestingNetwork):
    input_dims = (1, 3)
    def __init__(self, cfg, name=None):
        super().__init__(cfg, name=name)
        self.cfg = cfg

    def __call__(self, x, analysis, debug):
        x = hk.Linear(self.cfg['linear'], name="layer1", with_bias=False)(x)
        x = hk.Linear(self.cfg['linear'], name="layer2", with_bias=False)(x)
        return x

# probe1 = create_probe('layer2', 'w', 'gradients', execution_order='before', intercept_fn=lambda x: x*2)
probe1 = create_probe('layer1', 'r', 'params', execution_order='before', intercept_fn=lambda x: x*2)
probe2 = create_probe(hk.Linear, 'r', 'activations', execution_order='before')
probes = [probe1, probe2]
# probes = [probe2]
model = HaikuAutoInit(cfg, SimpleLinear, probes=probes)
rng_key = hk.PRNGSequence(12392)
inp = jax.random.normal(next(rng_key), (2, 3))
out, state = model(inp)

print('\n grad pass \n')
print(out['activations'])
losses, other, (frozen_params, trainable_params), (frozen_state, trainable_state), opt_state = model.train(inp)

print(other.keys())
[print(k, v['w'].shape) for k, v in other['grad_probes'].items()]
print(jnp.sum(jnp.abs(list(other['grad_probes'].values())[0]['w'])))