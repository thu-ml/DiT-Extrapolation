import torch
import triton
import triton.language as tl

from .quant_per_block import per_block_int8

from .attn_qk_int8_per_block import forward as attn_false
from .flashattention import forward as fp16_attn
from typing import Any, List, Literal, Optional, Tuple, Union




def sage_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    tensor_layout: str ="HND", 
    is_causal=False, 
    sm_scale: Optional[float] = None, 
    smooth_k: bool =True,     
    xpos_xi: tl.constexpr = 0.9999934149894527,
    flags = None,
    block_bias = None,
    sigmoid_a: float = 1.0,
    alpha_xpos_xi: float = 0.97,
    beta_xpos_xi: float = 0.8,
    decay_mask = None,
    sink_width: int = 4,
    window_width: int = 16,
    multi_factor: Optional[float] = None,
    entropy_factor: Optional[float] = None,
    **kwargs
) -> torch.Tensor:
    """

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.
    
    flags: torch.Tensor, dtype = torch.int32
       modify the logits in attention accodrding to value in flags.
       shape: [batch_size, head_dim]

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - All tensors must be on the same cuda device.
    """
    assert tensor_layout == 'HND'
    b,h = q.shape[0],q.shape[1]
    if flags == None:
        flags = torch.zeros([b,h], dtype=torch.int32, device=q.device)
    
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Input tensors must be in dtype of torch.float16, torch.bfloat16, or torch.float32."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    headdim = q.size(-1)
    assert headdim in [64, 96, 128], "headdim should be in [64, 96, 128]."

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        k -= km
    else:
        km = None

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if headdim == 96:
        raise NotImplementedError
    
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q, k, sm_scale=sm_scale, tensor_layout=tensor_layout, BLKQ=128, BLKK=128
    )

    if is_causal:
        raise NotImplementedError
    else:
        o = attn_false(q_int8, k_int8, v, flags, block_bias, decay_mask, q_scale, k_scale, 
            tensor_layout=tensor_layout, output_dtype=dtype, xpos_xi=xpos_xi, sigmoid_a=sigmoid_a, 
            alpha_xpos_xi=alpha_xpos_xi, beta_xpos_xi=beta_xpos_xi,
            BLOCK_M=128, BLOCK_N=128,
            sink_width=sink_width,
            window_width=window_width,
            multi_factor=multi_factor,
            entropy_factor=entropy_factor,
        )

    return o
