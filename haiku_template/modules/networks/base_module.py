import abc
from functools import partial
import logging

import haiku as hk
from haiku_template.modules.utils import split_treemap, rename_treemap_branches

# Standard haiku forward function
def forward_fn(x, training, net, cfg, analysis=False):
    return net(cfg)(x, training, analysis)

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
    

class HaikuAutoInit(object):
    # This class will automatically initalize itself with a standard 
    # forward function etc
    # Also implements naming conventions that allow for easy parameter loading

    def __init__(self, cfg, network_class):
        assert isinstance(network_class, AbstractNetwork), "network_class must subclass AbstractNetwork"
        assert 'seed' in cfg, "Must specify seed in base-level config (cfg interpolation is fine)"

        # Initialize the module
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        self.network = hk.transform_with_state(partial(forward_fn, net=network_class, cfg=cfg))

    def __call__(self, x, training=True, analysis=False):
        out = self.network(x, training, analysis)
        return self.network(x, training, analysis)
    
    def load(self, params):
        # Rename loaded model if needed 
        loaded_params, loaded_state = loaded_model
        if override_param_matching_tuples is not None:
            rename_tuples = override_param_matching_tuples
        else:
            rename_tuples = model_class.pretrain_param_matching_tuples if hasattr(model_class, 'pretrain_param_matching_tuples') else []

        loaded_params = rename_treemap_branches(loaded_params, rename_tuples)
        loaded_state = rename_treemap_branches(loaded_state, rename_tuples) 

        # Now split up model by relevant parts for training vs loading 
        if override_pretrain_partition_str is not None:
            pretrain_partition_string = override_pretrain_partition_str
        elif cfg_model.training.get('param_load_match', None):
            pretrain_partition_string = cfg_model.training['param_load_match']
        else:
            pretrain_partition_string = model_class.pretrain_partition_string if hasattr(model_class, 'pretrain_partition_string') else None # name of submodule whose weights are overwritten during loading
        

        trainable_params, trainable_state, loaded_params, loaded_state = split_treemap(trainable_params, trainable_state, 
                                                               loaded_params, loaded_state, pretrain_partition_string)
        
        self.params = (loaded_params, trainable_params)
        self.net_state = (loaded_state, trainable_state)
        logging.info(f"Model has params: {hk.data_structures.tree_size(trainable_params)} trainable and {hk.data_structures.tree_size(loaded_params) if loaded_params is not None else 0} frozen")






