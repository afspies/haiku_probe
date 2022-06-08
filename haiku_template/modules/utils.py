import haiku as hk


# -- Parameter Combination Functions -- #
def rename_treemap_branches(params, rename_tuples):
    """ Takes a haiku treemap datastructured (e.g. model params or state) and a list
    of the form [('old branch subname','new branch subname'), ...]
    
    Returns tree with renamed branches
    """

    if params is not None and len(rename_tuples) > 0: # Loaded model may have no associated state
        params = hk.data_structures.to_mutable_dict(params)
        initial_names = list(params.keys())
        for layer_name in initial_names:
            mapped_name = layer_name
            for (old_name, new_name) in rename_tuples:
                mapped_name = mapped_name.replace(old_name, new_name)
            params[mapped_name] = params[layer_name]
            if mapped_name != layer_name:
                params.pop(layer_name)
    return params


def split_treemap(trainable_params, trainable_state, loaded_params, loaded_state, partition_string=None):
    if loaded_params is not None:
        if partition_string is not None:
            if type(partition_string) is str:
                partition_string = [partition_string]
            # NOTE doesn't support fine-tuning - i.e. loaded_params = Frozen if we are partitioning
            trainable_params, _ = hk.data_structures.partition(
                                lambda m, n, p: any([ps in m for ps in partition_string]), trainable_params)
            trainable_state, _ = hk.data_structures.partition(
                                lambda m, n, p: any([ps in m for ps in partition_string]), trainable_state)
            loaded_params, _ = hk.data_structures.partition(
                                lambda m, n, p: any([ps not in m for ps in partition_string]), loaded_params)
            loaded_state, _ = hk.data_structures.partition(
                                lambda m, n, p: any([ps not in m for ps in partition_string]), loaded_state)
        else: # NOTE This assumes resuming from a checkpoint, but no option for pure testing  
            trainable_params = loaded_params
            trainable_state = loaded_state
            loaded_params = None
            loaded_state = None
    return trainable_params, trainable_state, loaded_params, loaded_state


#-- Loading Functions -- #
def load_params(path, load_from_step=None):
    if load_from_step is None: # we'll take the one trained for the longest    
        load_from_step = sorted(map(lambda x: int(x.split('_')[1][:-4]), os.listdir(path/'models')))[-1]
    
    # Load parameters
    with open(path/'models'/f'params_{load_from_step}.pkl', 'rb') as f:
        params = pickle.load(f)
    return params

def load_config(path):
    # Manually load config
    # Initialize - must use relative path to cwd because hydra says so
    initialize(config_path=os.path.relpath((path/'.hydra'), start=Path(os.path.realpath(__file__)).parent))
    return compose(config_name="config")

def load_saved_params(path, load_from_step=None):
    path = Path(path)
    cfg = load_config(path)
    params = load_params(path, load_from_step=load_from_step)
    return cfg, params

def load_saved_model(path):
    path = Path(path)
    cfg, (params, state) = load_saved_params(path)
    model_class = ModelClasses[cfg.model.name]
    return cfg, model_class, params, state
