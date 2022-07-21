import abc
from functools import partial
import logging

import optax
import haiku as hk
import jax
import jax.numpy as jnp
from haiku_probe.modules.utils import split_treemap, rename_treemap_branches
from haiku_probe.modules.prober import probe_manager, GradientProbe
import contextlib
from treeo import map as treemap


# Standard haiku forward function with prober functionality added
def forward_fn(x, training, analysis, net=None, cfg=None, prober=None):
    out = {}
    with prober(out) if prober is not None else contextlib.nullcontext(): 
        net_out = net(cfg)(x, training, analysis)
        if not isinstance(net_out, dict):
            net_out = {'out': net_out}
        out.update(net_out) 
    return out

class AbstractNetwork(hk.Module):
    """
    In order to be compatible with the trainer and tester,
    should inherit from this ABC and implement relevant functions.
    
    Analysis Components can return none if only training is desired
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, name=None):
        """
        cfg should allow dictionary-like item access 
        """
        super().__init__(name=name)
        self.cfg = cfg 
    
    @property
    @abc.abstractmethod
    def input_dims(self):
        pass

    # Training Components
    @staticmethod
    @abc.abstractmethod
    def get_optimizer(cfg):
        pass
    @staticmethod
    @abc.abstractmethod
    def get_loss(cfg):
        pass

    # Analysis Components
    @staticmethod
    @abc.abstractmethod
    def get_performance_tests(cfg):
        pass
    @staticmethod
    @abc.abstractmethod
    def get_visualizers(cfg):
        pass
    
    # NN Components
    def __call__(self, x, training=True, analysis=False): # Analysis flag determines completeness of outputs
        pass

class TestingNetwork(AbstractNetwork):
    def get_optimizer(_):
        pass
    def get_loss(_):
        pass
    def get_performance_tests(_):
        pass
    def get_visualizers(_):
        pass
    
class HaikuAutoInit(object):
    # This class will automatically initalize itself with a standard 
    # forward function etc
    # Also implements naming conventions that allow for easy parameter loading

    def __init__(self, cfg, network_class, probes=None):
        assert issubclass(network_class, AbstractNetwork), "network_class must subclass AbstractNetwork"
        assert 'seed' in cfg, "Must specify seed in base-level config (cfg interpolation is fine)"

        # Initialize the module
        self.rngseq = hk.PRNGSequence(cfg['seed'])
        prober = partial(probe_manager, probes)
        self.network = hk.transform_with_state(partial(forward_fn, net=network_class, cfg=cfg, prober=prober))
        self.grad_probes = list(filter(lambda x: isinstance(x, GradientProbe), probes))

        # jitted_init = jax.jit(partial(self.network.init, training=True, analysis=False))
        jitted_init = partial(self.network.init, training=True, analysis=False)
        trainable_params, trainable_state = jitted_init(next(self.rngseq), jnp.zeros(network_class.input_dims))

        self.params = trainable_params
        self.state = trainable_state

        # Initialize optimizer and loss function
        opt_init, self.opt_update =  optax.adam(1e-4) #model_class.get_optimizer(cfg_model.training)
        self.opt_state = opt_init(trainable_params)
        self.loss_fn = lambda x, y, _1, _2: (x**2).mean() #model_class.get_loss(cfg_model.training)
        # -y
    
    # @partial(jax.jit, static_argnums=(0,))x
    def __call__(self, x, training=True, analysis=False):
        out = self.network.apply(self.params, self.state, next(self.rngseq), x, training, analysis)
        return out

    def train(self, batch):
        return self.update(None, self.params, None, self.state, next(self.rngseq), self.opt_state, batch)

    @partial(jax.jit, static_argnums=(0,)) 
    def update(self, frozen_params, trainable_params, frozen_state, trainable_state, rng_key, opt_state, batch):
        """Learning rule (stochastic gradient descent)."""
        train_grads, (losses, trainable_state, other) = jax.grad(self._loss, 1, has_aux=True)(frozen_params, trainable_params, 
                                                                                        frozen_state, trainable_state,
                                                                                        rng_key, batch)
        train_grads, probe_grads = self._apply_gradient_probes(self.grad_probes, train_grads)
        if len(self.grad_probes) > 0: other['grad_probes'] = probe_grads
        updates, opt_state = self.opt_update(train_grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)
        # other = (other, train_grads)
        return losses, other, (frozen_params, trainable_params), (frozen_state, trainable_state), opt_state

    def _loss(self, frozen_params, trainable_params, frozen_state, trainable_state, rng_key, batch):
        params = trainable_params if frozen_params is None else hk.data_structures.merge(frozen_params, trainable_params)
        state = trainable_state if frozen_state is None else hk.data_structures.merge(frozen_state, trainable_state)
        x, state = self.network.apply(params, state, rng_key, batch, training=True, analysis=False)
        loss = self.loss_fn(x['out'], batch, rng_key, params)
        #other should be a dict of k: values where values get logged to wandb
        return loss, (loss, state, x) #if 'other' in x.keys() else N


    # @staticmethod
    # def filter_by_layer(current_layer): #, target_layer): 
    #     """
    #     Function acts as a filter for the "is_leaf" argument of the jax treemap function
    #         current_layer is the pytree passed by the map function
    #         target_layer can be a string or layermodule
    #     """
    #     fn = lambda module_name, name, value: 2 * value if name == 'w' else value
    #     if isinstance(current_layer, dict):
    #         if 'simple_linear/layer2' in current_layer.keys(): # Target Layer
    #             return True # Treat this as a leaf - the children include our target
    #     return False # Need to recurse deeper

    def _apply_gradient_probes(self, probes, grads):
        # Shall I apply all probes across all grads, or all probes for each grad?
        # All probes across all grads for easier accumulation?
        out_grads = None
        for probe in probes:
            if isinstance(probe, GradientProbe):
                out_grads = map_and_filter(probe, grads)
        return grads, out_grads

from haiku.data_structures import traverse, to_haiku_dict
from collections import defaultdict
def map_and_filter(fn, structure):
    """Maps a function to an input structure accordingly.
    >>> params = {'linear': {'w': 1.0, 'b': 2.0}}
    >>> fn = lambda module_name, name, value: 2 * value if name == 'w' else value
    >>> hk.data_structures.map(fn, params)
    {'linear': {'b': 2.0, 'w': 2.0}}
    Note: returns a new structure not a view.
    Args:
        fn: criterion to be used to map the input data.
        The ``fn`` argument is expected to be a function taking as inputs the
        name of the module, the name of a given entry in the module data bundle
        (e.g. parameter name) and the corresponding data, and returning a new
        value.
        structure: Haiku params or state data structure to be mapped.
    Returns:
        All the input parameters or state as mapped by the input fn.
    """
    out = defaultdict(dict)
    for module_name, name, value in traverse(structure):
        mapped_out = fn(module_name, name, value)
        if mapped_out is not None:
            out[module_name][name] = mapped_out
    return to_haiku_dict(out)
# ---------------------------------------------------------------------------------------------------------------------
