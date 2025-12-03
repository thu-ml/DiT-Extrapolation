from typing import Optional, Tuple

import torch
import torch.backends.cuda as cuda_backend
import torch.nn.functional as F
from torch.backends.cuda import SDPAParams
from torch.nn.attention import SDPBackend

aten = torch.ops.aten


def cannot_use_attention_backend(*args, **kwargs):
    return False


can_use_flash_attention = getattr(cuda_backend, "can_use_flash_attention", cannot_use_attention_backend)
can_use_efficient_attention = getattr(cuda_backend, "can_use_efficient_attention", cannot_use_attention_backend)
can_use_cudnn_attention = getattr(cuda_backend, "can_use_cudnn_attention", cannot_use_attention_backend)

# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:

    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


def flash_attention_forward_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return aten._scaled_dot_product_flash_attention(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )[:2]


def efficient_attention_forward_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return aten._scaled_dot_product_efficient_attention(
        query,
        key,
        value,
        attn_bias=attn_mask,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )[:2]


def cudnn_attention_forward_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return aten._scaled_dot_product_cudnn_attention(
        query,
        key,
        value,
        attn_bias=attn_mask,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )[:2]


def _attention_forward_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = query.device
    assert device.type == "cuda", "Must be CUDA tensors"
    assert torch.version.hip is None, "HIP is not supported"
    params_init_args = [
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
    ]
    if hasattr(SDPAParams, "enable_gqa"):
        params_init_args.append(False)
    params = SDPAParams(*params_init_args)
    dispatches = [
        ["FLASH_ATTENTION", can_use_flash_attention, flash_attention_forward_with_lse],
        ["EFFICIENT_ATTENTION", can_use_efficient_attention, efficient_attention_forward_with_lse],
        ["CUDNN_ATTENTION", can_use_cudnn_attention, cudnn_attention_forward_with_lse],
    ]
    if hasattr(torch._C, "_get_sdp_priority_order"):
        priority_order = torch._C._get_sdp_priority_order()
        priority_order = [str(SDPBackend(i)).split(".")[-1] for i in priority_order]
    else:
        device_capability = torch.cuda.get_device_capability(device.index)
        if device_capability >= (9, 0):
            priority_order = ["CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"]
        else:
            priority_order = ["FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION"]
    dispatches = [dispatch for dispatch in dispatches if dispatch[0] in priority_order]
    dispatches = sorted(dispatches, key=lambda x: priority_order.index(x[0]))
    for dispatch in dispatches:
        if dispatch[1](params):
            return dispatch[2](
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )
    raise NotImplementedError


@_torch_custom_op_wrapper("para_attn::attention_forward_with_lse", mutates_args=(), device_types=("cpu", "cuda"))
def attention_forward_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _attention_forward_with_lse(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@_torch_register_fake_wrapper("para_attn::attention_forward_with_lse")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _attention_forward_with_lse(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@_torch_custom_op_wrapper("para_attn::attention_forward", mutates_args=(), device_types=("cpu", "cuda"))
def attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@_torch_register_fake_wrapper("para_attn::attention_forward")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _attention_forward_sparse_kv(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
):
    if attn_mask is None:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    assert attn_mask.dtype == torch.bool, "attn_mask must be a boolean tensor"
    assert not is_causal, "is_causal is not supported with sparse kv"

    s_kv = key.shape[-2]
    indices = torch.arange(s_kv, device=key.device)

    outputs = []
    for q, k, v, m in zip(query, key, value, attn_mask):
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        while m.ndim > 1:
            m = m[0]
        indices_ = indices[m]
        k = k[..., indices_, :]
        v = v[..., indices_, :]
        outputs.append(
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        )
        del q, k, v, m, indices_

    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


@_torch_custom_op_wrapper("para_attn::attention_forward_sparse_kv", mutates_args=(), device_types=("cpu", "cuda"))
def attention_forward_sparse_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return _attention_forward_sparse_kv(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@_torch_register_fake_wrapper("para_attn::attention_forward_sparse_kv")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return _attention_forward_sparse_kv(
        query,
        key,
        value,
        # attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
