'''
When the attn block and the sparse block are the same or different, the attn_output and block_avg results are consistent (apart from a slight numerical difference).
'''

import triton
import triton.language as tl
import math
import torch

@triton.jit
def _attn_fwd_search_1(Q, K, V, sm_scale, M, Out, Block_avg, #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_bz, stride_bh, stride_bm, stride_bn,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              SPARSE_BLOCK: tl.constexpr,
              B_SHAPE_M: tl.constexpr,
              B_SHAPE_N: tl.constexpr,
              STAGE: tl.constexpr  #
              ):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    # tl.static_assert(SPARSE_BLOCK <= BLOCK_M)
    # tl.static_assert(SPARSE_BLOCK <= BLOCK_N)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # tmp_block_avg = tl.zeros([B_SHAPE_N], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)


    # epilogue
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1) 
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)

        p = p.to(q.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    
    K_block_ptr_origin = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    lo, hi = 0, N_CTX
    K_block_ptr_origin = tl.advance(K_block_ptr_origin, (0, lo))
    # loop over k, v and update accumulator
    # idx = tl.arange(0, B_SHAPE_N)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        block_j = start_n // BLOCK_N
        # -- compute qk ----
        k2 = tl.load(K_block_ptr_origin)
        qk2 = tl.dot(q, k2)
        qk2_scaled = qk2 * qk_scale
        p2 = tl.exp2(qk2_scaled - m_i[:, None])
        # ---- sum them up to get the sub-block sum
        p2_sum = tl.sum(p2)  # single scalar across BLOCK_M,BLOCK_N
        # tmp_block_avg = tl.where(idx == block_j, p2_sum, tmp_block_avg)
        
        K_block_ptr_origin = tl.advance(K_block_ptr_origin, (0, BLOCK_N))
        block_avg_ptr = (
            Block_avg
            + off_z * stride_bz 
            + off_h * stride_bh
            + start_m * stride_bm
            + block_j * stride_bn
        )
        tl.store(block_avg_ptr, p2_sum)




@triton.jit
def _attn_fwd_search_2(Q, K, V, sm_scale, M, Out, Block_avg, #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_bz, stride_bh, stride_bm, stride_bn,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              SPARSE_BLOCK: tl.constexpr,
              B_SHAPE_M: tl.constexpr,
              B_SHAPE_N: tl.constexpr,
              STAGE: tl.constexpr  #
              ):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    # tl.static_assert(SPARSE_BLOCK <= BLOCK_M)
    # tl.static_assert(SPARSE_BLOCK <= BLOCK_N)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # tmp_block_avg = tl.zeros([B_SHAPE_N], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)


    # epilogue
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1) 
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)

        p = p.to(q.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    
    K_block_ptr_origin = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    lo, hi = 0, N_CTX
    K_block_ptr_origin = tl.advance(K_block_ptr_origin, (0, lo))
    # loop over k, v and update accumulator
    # idx = tl.arange(0, B_SHAPE_N)
    # lse_final = tl.load(m_ptrs)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        block_j = start_n // BLOCK_N
        # -- compute qk ----
        k2 = tl.load(K_block_ptr_origin)
        qk2 = tl.dot(q, k2) * qk_scale 
        p2 = tl.exp2(qk2 - m_i[:, None])



        # Initialize output tensor
        num_blocks_m = BLOCK_M // SPARSE_BLOCK
        num_blocks_n = BLOCK_N // SPARSE_BLOCK

        p2 = tl.reshape(p2, (2, SPARSE_BLOCK, 2, SPARSE_BLOCK))
        p2_sum1 = tl.sum(p2, axis = 1)
        p2_sum = tl.sum(p2_sum1, axis = 2)
        K_block_ptr_origin = tl.advance(K_block_ptr_origin, (0, BLOCK_N))
        block_avg_ptr = (
            Block_avg
            + off_z * stride_bz 
            + off_h * stride_bh
            + start_m * stride_bm * num_blocks_m 
            + block_j * stride_bn * num_blocks_n
            + tl.arange(0, 2)[:, None] * stride_bm
            + tl.arange(0, 2)[None,:]
        )
        tl.store(block_avg_ptr, p2_sum)






@triton.jit
def _attn_fwd_search_4(Q, K, V, sm_scale, M, Out, Block_avg, #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_bz, stride_bh, stride_bm, stride_bn,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              SPARSE_BLOCK: tl.constexpr,
              B_SHAPE_M: tl.constexpr,
              B_SHAPE_N: tl.constexpr,
              STAGE: tl.constexpr  #
              ):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    # tl.static_assert(SPARSE_BLOCK <= BLOCK_M)
    # tl.static_assert(SPARSE_BLOCK <= BLOCK_N)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # tmp_block_avg = tl.zeros([B_SHAPE_N], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)


    # epilogue
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1) 
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)

        p = p.to(q.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    
    K_block_ptr_origin = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    lo, hi = 0, N_CTX
    K_block_ptr_origin = tl.advance(K_block_ptr_origin, (0, lo))
    # loop over k, v and update accumulator
    # idx = tl.arange(0, B_SHAPE_N)
    # lse_final = tl.load(m_ptrs)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        block_j = start_n // BLOCK_N
        # -- compute qk ----
        k2 = tl.load(K_block_ptr_origin)
        qk2 = tl.dot(q, k2) * qk_scale 
        p2 = tl.exp2(qk2 - m_i[:, None])



        # Initialize output tensor
        num_blocks_m = BLOCK_M // SPARSE_BLOCK
        num_blocks_n = BLOCK_N // SPARSE_BLOCK

        p2 = tl.reshape(p2, (4, SPARSE_BLOCK, 4, SPARSE_BLOCK))
        p2_sum1 = tl.sum(p2, axis = 1)
        p2_sum = tl.sum(p2_sum1, axis = 2)
        K_block_ptr_origin = tl.advance(K_block_ptr_origin, (0, BLOCK_N))
        block_avg_ptr = (
            Block_avg
            + off_z * stride_bz 
            + off_h * stride_bh
            + start_m * stride_bm * num_blocks_m 
            + block_j * stride_bn * num_blocks_n
            + tl.arange(0, 4)[:, None] * stride_bm
            + tl.arange(0, 4)[None,:]
        )
        tl.store(block_avg_ptr, p2_sum)


def _flash_attn_triton_search(q, k, v, causal = False, sm_scale = None, BLOCK_M = 128, BLOCK_N = 128, sparse_block_size = 128):
        ### q.shape = [B, H, C, D]
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        B, H, C, D = q.shape
        C_ROUNDED = (C + sparse_block_size - 1) // sparse_block_size * sparse_block_size
        if sm_scale == None:
            sm_scale = 1.0 / math.sqrt(HEAD_DIM_Q)
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        # Tuning for AMD target
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        block_avg = torch.empty(
            (
                B,
                H,
                C_ROUNDED // sparse_block_size,
                C_ROUNDED // sparse_block_size,
            ),
            device=q.device,
            dtype=torch.float32,
        )
        # print(f"block_avg stride: {block_avg.stride(0)}, {block_avg.stride(1)}, {block_avg.stride(2)}, {block_avg.stride(3)}")
        if BLOCK_M == BLOCK_N and BLOCK_M == sparse_block_size: 
            _attn_fwd_search_1[grid](
                q, k, v, sm_scale, M, o, block_avg, #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                block_avg.stride(0), block_avg.stride(1), block_avg.stride(2), block_avg.stride(3),  #
                q.shape[0], q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K, 
                BLOCK_M = BLOCK_M, #
                BLOCK_N = BLOCK_N,
                SPARSE_BLOCK = sparse_block_size,
                B_SHAPE_M = block_avg.shape[2],  #
                B_SHAPE_N = block_avg.shape[3],  #
                STAGE=stage,  #
                num_warps = 4,
                num_stages = 1,)
        elif BLOCK_M // sparse_block_size == 2:
            _attn_fwd_search_2[grid](
                q, k, v, sm_scale, M, o, block_avg, #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                block_avg.stride(0), block_avg.stride(1), block_avg.stride(2), block_avg.stride(3),  #
                q.shape[0], q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K, 
                BLOCK_M = BLOCK_M, #
                BLOCK_N = BLOCK_N,
                SPARSE_BLOCK = sparse_block_size,
                B_SHAPE_M = block_avg.shape[2],  #
                B_SHAPE_N = block_avg.shape[3],  #
                STAGE=stage,  #
                num_warps = 4,
                num_stages = 1,)
        elif BLOCK_M // sparse_block_size == 4:
            _attn_fwd_search_4[grid](
                q, k, v, sm_scale, M, o, block_avg, #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                block_avg.stride(0), block_avg.stride(1), block_avg.stride(2), block_avg.stride(3),  #
                q.shape[0], q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K, 
                BLOCK_M = BLOCK_M, #
                BLOCK_N = BLOCK_N,
                SPARSE_BLOCK = sparse_block_size,
                B_SHAPE_M = block_avg.shape[2],  #
                B_SHAPE_N = block_avg.shape[3],  #
                STAGE=stage,  #
                num_warps = 4,
                num_stages = 1,)
        
        
        return o, M, block_avg





# def torch_flash_attn_with_sparse_block(q, k, v, sm_scale, BLOCK_M, BLOCK_N, SPARSE_BLOCK):
#     """
#     PyTorch implementation of Flash Attention with block_avg calculation using SPARSE_BLOCK.

#     Args:
#         q, k, v: Tensors with shape [B, H, N_CTX, HEAD_DIM].
#         sm_scale: Scaling factor for QK product.
#         BLOCK_M, BLOCK_N: Block sizes for Flash Attention computation.
#         SPARSE_BLOCK: Block size for block_avg calculation.

#     Returns:
#         o: Output tensor with shape [B, H, N_CTX, HEAD_DIM].
#         block_avg: Block sum tensor with SPARSE_BLOCK x SPARSE_BLOCK blocks.
#     """
#     B, H, N_CTX, HEAD_DIM = q.shape

#     # Safe softmax calculation
#     attn_score = ((q @ k.transpose(-1, -2)) * sm_scale).to(torch.float32) # [B, H, N_CTX, N_CTX]
#     attn_score_max = attn_score.max(dim=-1, keepdim=True).values  # [B, H, N_CTX, 1]
#     attn_score = attn_score - attn_score_max  # Stabilize

#     # Softmax normalization
#     attn_score_exp = attn_score.exp2().to(torch.float32)   # [B, H, N_CTX, N_CTX]
#     attn_score_exp_sum = attn_score_exp.sum(dim=-1, keepdim=True).to(torch.float32)   # [B, H, N_CTX, 1]
#     attn_weight = attn_score_exp / attn_score_exp_sum.to(torch.float32)   # Safe softmax result

#     # Reshape to sparse block dimensions
#     attn_weight = attn_weight.reshape(
#         B, H, SPARSE_BLOCK, N_CTX // SPARSE_BLOCK, SPARSE_BLOCK, N_CTX // SPARSE_BLOCK
#     )  # [B, H, SB, n_blocks_m, SB, n_blocks_n]

#     # Compute block_avg by summing over SPARSE_BLOCK dimensions
#     block_avg = attn_weight.sum(dim=(2, 4))  # [B, H, n_blocks_m, n_blocks_n]

#     # Output tensor computation
#     # o = attn_weight @ v.reshape(B, H, SPARSE_BLOCK, -1, HEAD_DIM).permute(0, 1, 3, 2, 4)  # Flash Attention output
#     # o = o.reshape(B, H, N_CTX, HEAD_DIM)  # Restore original shape

#     return 0, block_avg