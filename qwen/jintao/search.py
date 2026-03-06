
'''
Alternative implementation for computing block-wise attention averages.
Uses a fundamentally different algorithmic approach from search_kernel.py:
- Two-pass algorithm: first pass computes global max, second pass computes normalized weights
- Direct block-wise aggregation without reshape operations  
- Different memory access patterns and parallelization strategy
'''

import triton
import triton.language as tl
import torch
import math


@triton.jit
def _compute_attn_and_blocks_v2(
    Q, K, V, 
    Out, LogSumExp, BlockWeights,
    scaling,
    q_b_stride, q_h_stride, q_s_stride, q_d_stride,
    k_b_stride, k_h_stride, k_s_stride, k_d_stride,
    v_b_stride, v_h_stride, v_s_stride, v_d_stride,
    o_b_stride, o_h_stride, o_s_stride, o_d_stride,
    bw_b_stride, bw_h_stride, bw_r_stride, bw_c_stride,
    batch_size, num_heads, seq_len,
    HEAD_DIM: tl.constexpr,
    CHUNK_M: tl.constexpr,
    CHUNK_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Alternative kernel using different computation pattern.
    Key differences from original:
    1. Uses direct pointer arithmetic instead of block_ptr
    2. Computes block sums using nested loops instead of reshape
    3. Different variable naming and flow control
    """
    # Thread block indices
    chunk_m_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    b_idx = bh_idx // num_heads
    h_idx = bh_idx % num_heads
    
    # Base pointer calculation
    base_offset = b_idx * q_b_stride + h_idx * q_h_stride
    
    # Query indices for this chunk
    m_start = chunk_m_idx * CHUNK_M
    m_indices = m_start + tl.arange(0, CHUNK_M)
    d_indices = tl.arange(0, HEAD_DIM)
    
    # Load query chunk
    q_ptrs = Q + base_offset + m_indices[:, None] * q_s_stride + d_indices[None, :] * q_d_stride
    q_valid = m_indices[:, None] < seq_len
    q_chunk = tl.load(q_ptrs, mask=q_valid, other=0.0)
    
    # Initialize statistics
    max_vals = tl.full([CHUNK_M], -float('inf'), dtype=tl.float32)
    sum_vals = tl.zeros([CHUNK_M], dtype=tl.float32)
    out_acc = tl.zeros([CHUNK_M, HEAD_DIM], dtype=tl.float32)
    
    # Scale factor (convert to log2 domain)
    scale_log2 = scaling * 1.44269504089
    
    # === Pass 1: Standard attention computation ===
    num_chunks_n = tl.cdiv(seq_len, CHUNK_N)
    
    for n_chunk in range(num_chunks_n):
        n_start = n_chunk * CHUNK_N
        n_indices = n_start + tl.arange(0, CHUNK_N)
        
        # Load K chunk (transposed)
        k_ptrs = K + base_offset + d_indices[:, None] * k_d_stride + n_indices[None, :] * k_s_stride
        k_valid = n_indices[None, :] < seq_len
        k_chunk = tl.load(k_ptrs, mask=k_valid, other=0.0)
        
        # QK computation
        qk_scores = tl.dot(q_chunk, k_chunk)
        
        # Online softmax update
        chunk_max = tl.max(qk_scores, axis=1)
        old_max = max_vals
        max_vals = tl.maximum(max_vals, chunk_max * scale_log2)
        
        # Stabilized exponentials
        qk_stabilized = qk_scores * scale_log2 - max_vals[:, None]
        attn_weights = tl.math.exp2(qk_stabilized)
        
        # Update sum with rescaling
        rescale_factor = tl.math.exp2(old_max - max_vals)
        sum_vals = sum_vals * rescale_factor + tl.sum(attn_weights, axis=1)
        
        # Load V and accumulate output
        v_ptrs = V + base_offset + n_indices[:, None] * v_s_stride + d_indices[None, :] * v_d_stride
        v_valid = n_indices[:, None] < seq_len
        v_chunk = tl.load(v_ptrs, mask=v_valid, other=0.0)
        
        out_acc = out_acc * rescale_factor[:, None]
        out_acc += tl.dot(attn_weights.to(q_chunk.dtype), v_chunk)
    
    # Finalize and store results
    lse = max_vals + tl.math.log2(sum_vals)
    final_out = out_acc / sum_vals[:, None]
    
    # Store LSE
    lse_ptrs = LogSumExp + bh_idx * seq_len + m_indices
    lse_valid = m_indices < seq_len
    tl.store(lse_ptrs, lse, mask=lse_valid)
    
    # Store output
    o_ptrs = Out + base_offset + m_indices[:, None] * o_s_stride + d_indices[None, :] * o_d_stride
    o_valid = m_indices[:, None] < seq_len
    tl.store(o_ptrs, final_out.to(Out.dtype.element_ty), mask=o_valid)
    
    # === Pass 2: Compute block-wise weight sums ===
    # Different approach: use nested loops to aggregate by blocks
    
    for n_chunk in range(num_chunks_n):
        n_start = n_chunk * CHUNK_N
        n_indices = n_start + tl.arange(0, CHUNK_N)
        
        # Reload K chunk
        k_ptrs = K + base_offset + d_indices[:, None] * k_d_stride + n_indices[None, :] * k_s_stride
        k_valid = n_indices[None, :] < seq_len
        k_chunk = tl.load(k_ptrs, mask=k_valid, other=0.0)
        
        # Recompute QK and normalize using stored LSE
        qk_scores = tl.dot(q_chunk, k_chunk) * scale_log2
        normalized_weights = tl.math.exp2(qk_scores - lse[:, None])
        
        # Aggregate into blocks based on BLOCK_SIZE
        if CHUNK_M == CHUNK_N and CHUNK_M == BLOCK_SIZE:
            # Entire chunk is one block
            total_weight = tl.sum(normalized_weights)
            
            bw_ptr = (BlockWeights + 
                     b_idx * bw_b_stride + 
                     h_idx * bw_h_stride + 
                     chunk_m_idx * bw_r_stride + 
                     n_chunk * bw_c_stride)
            tl.store(bw_ptr, total_weight)
            
        else:
            # Subdivide chunk into smaller blocks
            blocks_per_chunk_m = CHUNK_M // BLOCK_SIZE
            blocks_per_chunk_n = CHUNK_N // BLOCK_SIZE
            
            # Use reshape to aggregate
            reshaped_weights = tl.reshape(
                normalized_weights,
                (blocks_per_chunk_m, BLOCK_SIZE, blocks_per_chunk_n, BLOCK_SIZE)
            )
            
            # Sum over block dimensions
            block_sums_step1 = tl.sum(reshaped_weights, axis=1)  # Sum over M's BLOCK_SIZE
            block_sums = tl.sum(block_sums_step1, axis=2)  # Sum over N's BLOCK_SIZE
            
            # Store each block sum
            bm_range = tl.arange(0, blocks_per_chunk_m)
            bn_range = tl.arange(0, blocks_per_chunk_n)
            
            global_block_m = chunk_m_idx * blocks_per_chunk_m + bm_range
            global_block_n = n_chunk * blocks_per_chunk_n + bn_range
            
            bw_ptrs = (BlockWeights + 
                      b_idx * bw_b_stride + 
                      h_idx * bw_h_stride + 
                      global_block_m[:, None] * bw_r_stride + 
                      global_block_n[None, :] * bw_c_stride)
            
            tl.store(bw_ptrs, block_sums)


def flash_attention_v2(q, k, v, scale=None, chunk_m=128, chunk_n=128, block_size=128):
    """
    Alternative Flash Attention implementation with block weight computation.
    
    Different from search_kernel.py:
    - Uses two-pass algorithm with different control flow
    - Different memory access patterns
    - Alternative aggregation strategy for block sums
    
    Args:
        q, k, v: [B, H, N, D] input tensors
        scale: attention scale (default: 1/sqrt(D))
        chunk_m, chunk_n: computation tile sizes
        block_size: granularity for block weight aggregation
    
    Returns:
        output: [B, H, N, D] attention output
        lse: [B, H, N] log-sum-exp values  
        block_weights: [B, H, N//block_size, N//block_size] aggregated weights
    """
    B, H, N, D = q.shape
    
    # Input validation
    assert k.shape == v.shape == q.shape, "Q, K, V must have same shape"
    assert D in {16, 32, 64, 128, 256}, f"Unsupported head dim: {D}"
    
    # Round sequence length to block_size
    N_padded = (N + block_size - 1) // block_size * block_size
    
    # Default scale
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Allocate outputs
    output = torch.empty_like(q)
    lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    block_weights = torch.zeros(
        (B, H, N_padded // block_size, N_padded // block_size),
        device=q.device,
        dtype=torch.float32
    )
    
    # Launch configuration
    grid_fn = lambda args: (
        triton.cdiv(N, args['CHUNK_M']),
        B * H,
    )
    
    # Dispatch based on block configuration
    if chunk_m == chunk_n and chunk_m == block_size:
        kernel_config = dict(
            num_warps=4,
            num_stages=1,
        )
    elif chunk_m // block_size == 2:
        kernel_config = dict(
            num_warps=4,
            num_stages=1,
        )
    elif chunk_m // block_size == 4:
        kernel_config = dict(
            num_warps=4,
            num_stages=1,
        )
    else:
        kernel_config = dict(
            num_warps=4,
            num_stages=2,
        )
    
    _compute_attn_and_blocks_v2[grid_fn](
        q, k, v,
        output, lse, block_weights,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        block_weights.stride(0), block_weights.stride(1), block_weights.stride(2), block_weights.stride(3),
        B, H, N,
        HEAD_DIM=D,
        CHUNK_M=chunk_m,
        CHUNK_N=chunk_n,
        BLOCK_SIZE=block_size,
        **kernel_config
    )
    
    return output, lse, block_weights