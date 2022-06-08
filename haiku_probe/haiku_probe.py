from pathlib import Path
import os
import numpy as np
import random

import jax.numpy as jnp
import torch

import subprocess
import os

# -- Misc. -- # 
def seed(seed=42):
    # -- Python & Numpy -- 
    random.seed(seed)
    np.random.seed(seed)
  
    # -- Pytorch --
    torch.manual_seed(seed)
    #   The following may slightly slow down training:
    #   torch.use_deterministic_algorithms(True)
    #   os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    #   torch may also require specifying seeded worker for dataloader

# This function should be called after all imports,
# in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2):
    """Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
                                              
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
    """
    # Get the list of GPUs via nvidia-smi
    smi_query_result = subprocess.check_output('nvidia-smi -q -d Memory | grep -A4 GPU', shell=True)
    # Extract the usage information
    gpu_info = smi_query_result.decode('utf-8').split('\n')
    gpu_info = list(filter(lambda info: 'Used' in info, gpu_info))
    gpu_info = [int(x.split(':')[1].replace('MiB', '').strip()) for x in gpu_info] # Remove garbage
    gpu_info = gpu_info[:min(max_gpus, len(gpu_info))] # Limit to max_gpus
    # Assign free gpus to the current process
    gpus_to_use = ','.join([str(i) for i, x in enumerate(gpu_info) if x < threshold_vram_usage])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
    print(f'Using GPUs {gpus_to_use}' if gpus_to_use else 'No free GPUs found')


def transform_to_jax(*args):
    out = []
    for arg in args:
        if isinstance(arg, dict): 
            out.append({k: jnp.array(v, dtype=jnp.float32) for k, v in arg.items()})
        else:
            out.append(jnp.array(arg, dtype=jnp.float32))
    return out