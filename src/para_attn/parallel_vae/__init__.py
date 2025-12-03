import torch.distributed as dist

import para_attn.primitives as DP


def init_parallel_vae_mesh(device_type=None, *, mesh=None):
    if mesh is not None:
        return mesh

    assert device_type is not None, "device must be provided if mesh is not provided"

    world_size = DP.get_world_size()

    return dist.init_device_mesh(device_type, (world_size,))
