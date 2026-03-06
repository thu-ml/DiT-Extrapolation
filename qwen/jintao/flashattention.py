import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, old_m, q, 
                    K_ptrs, V_ptrs,  
                    start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    N_CTX: tl.constexpr):
    lo, hi = 0, N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        k_mask = offs_n[None, :] < (N_CTX - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)

        m = offs_m[:, None]
        n = start_n + offs_n

        qk = tl.dot(q, k).to(tl.float32) # + 2.1 * tl.abs(m-n)

        local_m = tl.max(qk, 1) * qk_scale
        new_m = tl.maximum(old_m, local_m)
        qk = qk * qk_scale - new_m[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(old_m - new_m)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask = offs_n[:, None] < (N_CTX - start_n))
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)   
        old_m = new_m
        K_ptrs += BLOCK_N * HEAD_DIM
        V_ptrs += BLOCK_N * HEAD_DIM
    return acc, l_i

@triton.jit
def _attn_fwd(Q, K, V, Out,  
              stride_qz, stride_qh, stride_qm, stride_qk,  
              stride_kz, stride_kh, stride_kn, stride_kk,  
              Z, H, N_CTX,  
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr  
              ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * stride_kn
    V_ptrs = V + qvk_offset + offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk
    O_block_ptr = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = HEAD_DIM**-0.5 * 1.44269504  
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < N_CTX)
    acc, l_i = _attn_fwd_inner(acc, l_i, m_i, q, K_ptrs, V_ptrs,  
                                    start_m, qk_scale,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n, N_CTX 
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < N_CTX))


def forward(q, k, v):
    BLOCK_M = 128
    BLOCK_N = 64
    HEAD_DIM_K = k.shape[-1]
    o = torch.empty_like(q, dtype=torch.float16)
    stage = 1
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    if k.size(-1) == 64:
        Cw, Cs = 4, 2
    elif k.size(-1) == 128:
        Cw, Cs = 8, 2
    _attn_fwd[grid](
        q, k, v, o,  
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  
        q.shape[0], q.shape[1],  
        N_CTX=q.shape[2],  
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage,  
        num_warps=Cw,  
        num_stages=Cs)
    return o