import functools
from typing import List, Optional, Union

import torch
from diffusers import DiffusionPipeline, FluxTransformer2DModel

import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import UnifiedAttnMode


def parallelize_transformer(transformer: FluxTransformer2DModel, *, mesh=None):
    if getattr(transformer, "_is_parallelized", False):
        return transformer

    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        controlnet_block_samples: Optional[List[torch.Tensor]] = None,
        controlnet_single_block_samples: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = DP.get_assigned_chunk(timestep, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=-2, group=seq_mesh)
        img_ids = DP.get_assigned_chunk(img_ids, dim=-2)
        txt_ids = DP.get_assigned_chunk(txt_ids, dim=-2)
        if controlnet_block_samples is not None:
            controlnet_block_samples = [
                DP.get_assigned_chunk(sample, dim=0, group=batch_mesh) for sample in controlnet_block_samples
            ]
            controlnet_block_samples = [
                DP.get_assigned_chunk(sample, dim=-2, group=seq_mesh) for sample in controlnet_block_samples
            ]
            kwargs["controlnet_block_samples"] = controlnet_block_samples
        if controlnet_single_block_samples is not None:
            controlnet_single_block_samples = [
                DP.get_assigned_chunk(sample, dim=0, group=batch_mesh) for sample in controlnet_single_block_samples
            ]
            controlnet_single_block_samples = [
                DP.get_assigned_chunk(sample, dim=-2, group=seq_mesh) for sample in controlnet_single_block_samples
            ]
            kwargs["controlnet_single_block_samples"] = controlnet_single_block_samples

        with UnifiedAttnMode(mesh):
            output = original_forward(
                hidden_states,
                encoder_hidden_states,
                *args,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                **kwargs,
            )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = DP.get_complete_tensor(sample, dim=-2, group=seq_mesh)
        sample = DP.get_complete_tensor(sample, dim=0, group=batch_mesh)
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    transformer.forward = new_forward.__get__(transformer)

    transformer._is_parallelized = True

    return transformer


def parallelize_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    if not getattr(pipe, "_is_parallelized", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, **kwargs):
            if generator is None and getattr(self, "_is_parallelized", False):
                seed_t = torch.randint(0, torch.iinfo(torch.int64).max, [1], dtype=torch.int64, device=self.device)
                seed_t = DP.get_complete_tensor(seed_t, dim=0)
                seed_t = DP.get_assigned_chunk(seed_t, dim=0, idx=0)
                seed = seed_t.item()
                seed -= torch.iinfo(torch.int64).min
                generator = torch.Generator(self.device).manual_seed(seed)
            return original_call(self, *args, generator=generator, **kwargs)

        new_call._is_parallelized = True

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_parallelized = True

    if not shallow_patch:
        parallelize_transformer(pipe.transformer, **kwargs)

    return pipe
