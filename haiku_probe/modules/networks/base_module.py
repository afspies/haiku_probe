import abc
from functools import partial
import logging

import haiku as hk
import jax
import jax.numpy as jnp
from haiku_probe.modules.utils import split_treemap, rename_treemap_branches
import contextlib

# Standard haiku forward function
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
    

import optax
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

    # @partial(jax.jit, static_argnums=(0,)) 
    def update(self, frozen_params, trainable_params, frozen_state, trainable_state, rng_key, opt_state, batch):
        """Learning rule (stochastic gradient descent)."""
        train_grads, (losses, trainable_state, other) = jax.grad(self._loss, 1, has_aux=True)(frozen_params, trainable_params, 
                                                                                        frozen_state, trainable_state,
                                                                                        rng_key, batch)
        train_grads, write_grads = self._apply_gradient_probes(train_grads)
        if write_grads: other['grad_probes'] = write_grads
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

    def _apply_gradient_probes(self, probes, grads):
        jax.tree.filter
        for probe in probes:
            grads = probe(grads)
        return grads
# ---------------------------------------------------------------------------------------------------------------------

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
cfg = {'linear':16, 'seed':42}
class SimpleLinear(TestingNetwork):
    input_dims = (1, 3)
    def __init__(self, cfg, name=None):
        super().__init__(cfg, name=name)
        self.cfg = cfg
    def __call__(self, x, analysis, debug):
        return hk.Linear(self.cfg['linear'], with_bias=False)(x)

@contextlib.contextmanager
def probe_manager(probes, output):
    ctx_managers = []
    for probe in probes:
        if isinstance(probe, hk.GetterContext):
            ctx_managers.append(hk.custom_getter(probe(output)))
        elif isinstance(probe, hk.SetterContext):
            ctx_managers.append(hk.custom_setter(probe(output)))
        else:
            ctx_managers.append(hk.intercept_methods(probe(output)))

    with contextlib.ExitStack() as stack:
        for mgr in ctx_managers:
            stack.enter_context(mgr)
        try:
            yield [stack] # forward function is getting called in here
        finally:
            pass

def context_matched(user_context, prober_context):
    if isinstance(prober_context,  (hk.GetterContext, hk.SetterContext)): # Using Getter
        pass
    else: # Using Interceptor
        if prober_context.method_name != '__call__':
            return False

    # Targeting a specific layer
    if isinstance(user_context, str): 
        return user_context not in prober_context.full_name

    # Targeting all modules of a specific form 
    if issubclass(user_context, hk.Module): 
        return user_context == type(prober_context.module)

    # if hasattr(prober_context.module, 'name'):
        # print(f'\t interceptor on {prober_context.module.name} ', end=' ')

@partial(jax.custom_vjp, nondiff_argnums=(1,))
def print_f(x, fun):
    return fun(x)

def print_f_fwd(x, fun):
    print('fwd')
    return print_f(x, fun), x 

def print_f_bwd(fun, res, grad):
    x = res
    print('bwd', grad)
    return fun(x),

print_f.defvjp(print_f_fwd, print_f_bwd)

def general_interceptor(applied_function, user_context, ordering, # Apply Closure
                        next_f, args, kwargs, context): # Intercepted at runtime
    if not context_matched(user_context, context):
        return next_f(*args, **kwargs)
    
    print(args)

    if ordering == 'before':
        args, kwargs = jax.tree_map(partial(applied_function, context=context), (args, kwargs))
        x = args[0]
        x = print_f(x, lambda x: x)
        args = (x, *args[1:])
    elif ordering == 'after': 
        #! only able to read after atm
        #! Think this duplicates operations - automatic interceptor generation for next op is the best option
        #! But will be implementationally painful
        """
        Slightly complicated - have access to next interceptor and current op
        can call current op, but must skip next interceptor's op call
        """
        out = context.orig_method(*args, **kwargs)
        jax.tree_map(partial(applied_function, context=context), (out))

    # return print_f(args[0], next_f) #, *args[1:], **kwargs)
    return next_f(*args, **kwargs)
    
def create_probe(user_context, probe_type, target_type, intercept_fn=None, execution_order='before'):

    if target_type in ['params', 'state']:
        if probe_type == 'r':
            def read_probe(output):
                # Able to get weights and state, but not activations
                def param_getter(next_getter, value, context, out=None):
                    if context_matched(user_context, context):
                        out[context.full_name] = value.astype(jnp.float16)
                    return next_getter(value)
                return partial(param_getter, out=output)
            return read_probe
        if probe_type == 'w':
            pass
    
    if target_type == 'gradients':
        if probe_type == 'r':
            def read_probe(output):
                def param_setter(next_setter, value, orig_dtype, context, out=None):
                    if not hk.running_init():
                        if context_matched(user_context, context):
                            out[context.module.name] = value#.astype(jnp.float16)
                    return next_setter(*value)
                return partial(param_setter, out=output)
            return read_probe

        if probe_type == 'w':
            def write_probe(output):
                def param_setter(next_setter, value, orig_dtype, context, out=None):
                    if not hk.running_init():
                        if context_matched(user_context, context):
                            out[context.module.name] = value
                    return next_setter(*value)


    if target_type in ['activations']:
        if target_type == 'activations':
            if probe_type == 'r':
                def intercept_fn(x, out=None, context=None):
                    out[context.module.name] = x
                    return x
            if probe_type == 'w':
                assert intercept_fn is not None, "Must provide interceptor function if modifying activations"
        else:
            if probe_type == 'r':
                def intercept_fn(x, out=None, context=None):
                    if hasattr(x, 'grad'):
                        out[context.module.name] = x.grads
                    return x
            if probe_type == 'w':
                assert intercept_fn is not None, "Must provide interceptor function if modifying gradients"

        def interceptor(output): 
            interceptor = partial(general_interceptor, partial(intercept_fn, out=output), user_context, execution_order) 
            return interceptor
        return interceptor

    return 'sort your life out'

probes = create_probe(hk.Linear, 'r', 'gradients', execution_order='before')
prober = partial(probe_manager, [probes])
model = HaikuAutoInit(cfg, SimpleLinear, prober=prober)
rng_key = hk.PRNGSequence(12392)
inp = jax.random.normal(next(rng_key), (2, 3))
out, state = model(inp)

print('\n grad pass \n')
losses, other, (frozen_params, trainable_params), (frozen_state, trainable_state), opt_state = model.train(inp)
[print(k, v.shape) for k, v in other.items()]
exit()
jax.grad(out)
# print(jnp.equal(out['out'], out['linear']).all())


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






