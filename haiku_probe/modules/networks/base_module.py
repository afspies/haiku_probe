import abc
from functools import partial
import logging

import haiku as hk
import jax
import jax.numpy as jnp
from haiku_probe.modules.utils import split_treemap, rename_treemap_branches
import contextlib

# Standard haiku forward function
def forward_fn(x, training, analysis, net=None, cfg=None, prober=contextlib.nullcontext()):
    with prober(): 
        out = net(cfg)(x, training, analysis)
    return  out

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

    def __init__(self, cfg, network_class, prober=None):
        assert issubclass(network_class, AbstractNetwork), "network_class must subclass AbstractNetwork"
        assert 'seed' in cfg, "Must specify seed in base-level config (cfg interpolation is fine)"

        # Initialize the module
        self.rngseq = hk.PRNGSequence(cfg['seed'])
        self.network = hk.transform_with_state(partial(forward_fn, net=network_class, cfg=cfg, prober=prober))
        
        jitted_init = jax.jit(partial(self.network.init, training=True, analysis=False))
        trainable_params, trainable_state = jitted_init(next(self.rngseq), jnp.zeros(network_class.input_dims))

        self.params = trainable_params
        self.state = trainable_state
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x, training=True, analysis=False):
        out = self.network.apply(self.params, self.state, next(self.rngseq), x, training, analysis)
        return out 



import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
cfg = {'linear':16, 'seed':42}
class SimpleLinear(TestingNetwork):
    input_dims = (64, 3)
    def __init__(self, cfg, name=None):
        super().__init__(cfg, name=name)
        self.cfg = cfg
    def __call__(self, x, analysis, debug):
        return hk.Linear(self.cfg['linear'])(x)


@contextlib.contextmanager
def prober(readers, writers):
    ctx_managers = []
    for writer in writers:
        ctx_managers.append(hk.intercept_methods(writer))

    with contextlib.ExitStack() as stack:
        for mgr in ctx_managers:
            stack.enter_context(mgr)
        try:
            yield [stack] # forward function is getting called in here
        finally:
            print('Pass complete')

def my_interceptor(next_f, args, kwargs, context):
    if (type(context.module) is not hk.BatchNorm
        or context.method_name != "__call__"):
        print('called_intercetpro on', type(context.module))
        if hasattr(context.module, 'name'):
            print(f'\t interceptor on {context.module.name}')
        # We ignore methods other than BatchNorm.__call__.
        return next_f(*args, **kwargs)

    def cast_if_array(x):
        if isinstance(x, jnp.ndarray):
            x = x.astype(jnp.float32)
        return x

probes = my_interceptor
prober = partial(prober, None, [probes])
model = HaikuAutoInit(cfg, SimpleLinear, prober=prober)
rng_key = hk.PRNGSequence(12392)
out, state = model(jax.random.normal(next(rng_key), (28, 3)))
print(out.shape)


    # def load(self, params):
    #     # Rename loaded model if needed 
    #     loaded_params, loaded_state = loaded_model
    #     if override_param_matching_tuples is not None:
    #         rename_tuples = override_param_matching_tuples
    #     else:
    #         rename_tuples = model_class.pretrain_param_matching_tuples if hasattr(model_class, 'pretrain_param_matching_tuples') else []

    #     loaded_params = rename_treemap_branches(loaded_params, rename_tuples)
    #     loaded_state = rename_treemap_branches(loaded_state, rename_tuples) 

    #     # Now split up model by relevant parts for training vs loading 
    #     if override_pretrain_partition_str is not None:
    #         pretrain_partition_string = override_pretrain_partition_str
    #     elif cfg_model.training.get('param_load_match', None):
    #         pretrain_partition_string = cfg_model.training['param_load_match']
    #     else:
    #         pretrain_partition_string = model_class.pretrain_partition_string if hasattr(model_class, 'pretrain_partition_string') else None # name of submodule whose weights are overwritten during loading
        

    #     trainable_params, trainable_state, loaded_params, loaded_state = split_treemap(trainable_params, trainable_state, 
    #                                                            loaded_params, loaded_state, pretrain_partition_string)
        
    #     self.params = (loaded_params, trainable_params)
    #     self.net_state = (loaded_state, trainable_state)
    #     logging.info(f"Model has params: {hk.data_structures.tree_size(trainable_params)} trainable and {hk.data_structures.tree_size(loaded_params) if loaded_params is not None else 0} frozen")






