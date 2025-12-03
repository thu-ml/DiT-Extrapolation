import functools
from typing import List, Optional, Tuple, Union

import torch
from diffusers import CogVideoXTransformer3DModel, DiffusionPipeline

import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import UnifiedAttnMode


def parallelize_transformer(transformer: CogVideoXTransformer3DModel, *, mesh=None):
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
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        *args,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = DP.get_assigned_chunk(timestep, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=-2, group=seq_mesh)

        if image_rotary_emb is not None:
            temporal_size = hidden_states.shape[1]
            freqs_cos, freqs_sin = image_rotary_emb

            def get_rotary_emb_chunk(freqs):
                dim_thw = freqs.shape[-1]
                freqs = freqs.reshape(temporal_size, -1, dim_thw)
                freqs = DP.get_assigned_chunk(freqs, dim=-2, group=seq_mesh)
                freqs = freqs.reshape(-1, dim_thw)
                return freqs

            freqs_cos = get_rotary_emb_chunk(freqs_cos)
            freqs_sin = get_rotary_emb_chunk(freqs_sin)
            image_rotary_emb = (freqs_cos, freqs_sin)

        with UnifiedAttnMode(mesh):
            output = original_forward(
                hidden_states,
                encoder_hidden_states,
                *args,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
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

    original_patch_embed_forward = transformer.patch_embed.forward

    @functools.wraps(transformer.patch_embed.__class__.forward)
    def new_patch_embed_forward(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        *args,
        **kwargs,
    ):
        text_embeds = DP.get_complete_tensor(text_embeds, dim=-2, group=seq_mesh)
        image_embeds = DP.get_complete_tensor(image_embeds, dim=-2, group=seq_mesh)

        batch, num_frames, channels, height, width = image_embeds.shape
        seq_length = text_embeds.shape[-2]

        embeds = original_patch_embed_forward(
            text_embeds,
            image_embeds,
            *args,
            **kwargs,
        )

        text_embeds = embeds[:, :seq_length]
        image_embeds = embeds[:, seq_length:].reshape(batch, num_frames, -1, embeds.shape[-1])
        text_embeds = DP.get_assigned_chunk(text_embeds, dim=-2, group=seq_mesh)
        image_embeds = DP.get_assigned_chunk(image_embeds, dim=-2, group=seq_mesh)
        image_embeds = image_embeds.flatten(1, 2)
        embeds = torch.cat([text_embeds, image_embeds], dim=1)

        return embeds

    transformer.patch_embed.forward = new_patch_embed_forward.__get__(transformer)

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
