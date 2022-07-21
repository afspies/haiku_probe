
from multiprocessing import context
import os
import haiku as hk
import contextlib
import jax
from functools import partial


@contextlib.contextmanager
def probe_manager(probes, output):
    ctx_managers = []
    for probe in probes:
        if isinstance(probe, hk.GetterContext):
            ctx_managers.append(hk.custom_getter(probe(output)))
        elif isinstance(probe, hk.SetterContext):
            ctx_managers.append(hk.custom_setter(probe(output)))
        elif isinstance(probe, GradientProbe):
            continue
        else:
            ctx_managers.append(hk.intercept_methods(probe(output)))

    with contextlib.ExitStack() as stack:
        for mgr in ctx_managers:
            stack.enter_context(mgr)
        try:
            yield [stack] # forward function is getting called in here
        finally:
            pass

def match_param_names(probe_query, param_name):
    if probe_query in param_name:
        return True
    return False


def context_matched(user_context, prober_context):
    if isinstance(prober_context,  (hk.GetterContext, hk.SetterContext)): # Using Getter or Setter
        pass
    elif isinstance(prober_context, (hk.Module, str)): # Using gradient intereptor and targeting module
        pass
        # prober_context = str(hk.get_module(prober_context))
    else: # Using Interceptor
        if prober_context.method_name != '__call__':
            return False

    # Targeting a specific layer
    if isinstance(user_context, str): 
        if isinstance(prober_context, str):
            return match_param_names(user_context, prober_context)
        return user_context not in prober_context.full_name

    # Targeting all modules of a specific form 
    if issubclass(user_context, hk.Module): 
        # handle case where prober context is a hiaku module or a string 
        if isinstance(prober_context, str):
            user_context  = user_context.__name__.split('.')[-1].lower()
            return match_param_names(str(user_context), prober_context)

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

# print_f.defvjp(print_f_fwd, print_f_bwd)

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
    
class GradientProbe():
    def __init__(self, context, apply_fn) -> None:
        self.context = context
        self.apply_fn = apply_fn
    
    def __call__(self, module_name, name, value):
        """
        Filtering function can only decide whether to retur entire subtree early,
        so also need to match subtree name here
        
        module name is full name, e.g. parent/child/linear1 
        name is param leaf name ,e.g. w or b
        """
        # if isinstance(inp, dict) and context_matched(self.context, next(iter(inp.keys()))):
            # The map filtering stopped early because this is garbage
            # print('mismatched with context, expected', self.context, 'got', next(iter(inp.keys())))
            # return inp
        if context_matched(self.context, module_name):
        # print('happy match - probe got', inp)
            return self.apply_fn(value)
        
    

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
            # def read_probe(output):
            """
            Gradient functions will be mapped aross the jax tree of gradients
            They are provided with the name of the params to which they are being applied
            and can manipulate values in-place, or append to a returned tree datastructre (the third arg)
            """
            # params are :
            # hk.Linear, 'r', 'gradients', execution_order='before'
            # User context will be layer name or type
            def weight_update(weight):
                print('fish', weight.shape)
                return weight*2.0
            return GradientProbe(user_context, weight_update)


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






