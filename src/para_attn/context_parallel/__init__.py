import math

import torch.distributed as dist

import para_attn.primitives as DP


def init_context_parallel_mesh(
    device_type=None, *, mesh=None, max_batch_dim_size=None, max_ring_dim_size=None, max_ulysses_dim_size=None
):
    if mesh is not None:
        return mesh

    assert device_type is not None, "device must be provided if mesh is not provided"

    world_size = DP.get_world_size()
    if max_batch_dim_size is None:
        batch_dim_size = 1
    else:
        batch_dim_size = math.gcd(world_size, max_batch_dim_size)

    attn_world_size = world_size // batch_dim_size

    assert not (
        max_ring_dim_size is not None and max_ulysses_dim_size is not None
    ), "Only one of max_ulysses_dim_size and max_ring_dim_size can be set"

    if max_ulysses_dim_size is None:
        if max_ring_dim_size is None:
            ring_dim_size = 1
        else:
            ring_dim_size = math.gcd(attn_world_size, max_ring_dim_size)
        ulysses_dim_size = attn_world_size // ring_dim_size
    else:
        ulysses_dim_size = math.gcd(attn_world_size, max_ulysses_dim_size)
        ring_dim_size = attn_world_size // ulysses_dim_size

    mesh_shape = (batch_dim_size, ring_dim_size, ulysses_dim_size)
    return dist.init_device_mesh(device_type, mesh_shape, mesh_dim_names=("batch", "ring", "ulysses"))
