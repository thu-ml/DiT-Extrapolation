import contextlib

import torch
import torch.nn.functional as F

import para_attn
import para_attn.ops as para_attn_ops
from para_attn.utils import base_handle_torch_function, BaseTorchFunctionMode

try:
    import torch.distributed.tensor.experimental._attention as torch_ring_attention
except ImportError:
    torch_ring_attention = None


class SparseKVAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        dispatch_to_custom_ops=True,
    ):
        out = (
            para_attn_ops.attention_forward_sparse_kv
            if dispatch_to_custom_ops
            else para_attn_ops._attention_forward_sparse_kv
        )(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError("Backward pass for SparseKVAttnFunc is not implemented")


def sparse_kv_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    dispatch_to_custom_ops=True,
):
    return SparseKVAttnFunc.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        dispatch_to_custom_ops,
    )


class StructuredSparseAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        sparse_mask,
        sparse_range_query,
        sparse_range_key_value,
        print_attn_weight_means,
    ):
        if sparse_mask is None:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

        assert torch_ring_attention is not None, "StructuredSparseAttnFunc requires a newer version of PyTorch"

        assert query.ndim == 4, "query must have 4 dimensions, got {}".format(query.ndim)
        assert key.ndim == 4, "key must have 4 dimensions, got {}".format(key.ndim)
        assert value.ndim == 4, "value must have 4 dimensions, got {}".format(value.ndim)

        assert not is_causal, "is_causal is not supported in StructuredSparseAttnFunc"

        assert sparse_mask.ndim == 2, "sparse_mask must have 2 dimensions, got {}".format(sparse_mask.ndim)

        b, h, s_q, d_qk = query.shape
        _, _, s_kv, d_v = value.shape

        if sparse_range_query is None:
            sparse_range_query = (0, s_q)
        if sparse_range_key_value is None:
            sparse_range_key_value = (0, s_kv)

        assert (
            0 <= sparse_range_query[0] <= sparse_range_query[1] <= s_q
        ), f"sparse_range_query must be in [0, {s_q}], got {sparse_range_query}"
        assert (
            0 <= sparse_range_key_value[0] <= sparse_range_key_value[1] <= s_kv
        ), f"sparse_range_key_value must be in [0, {s_kv}], got {sparse_range_key_value}"

        assert (sparse_range_query[1] - sparse_range_query[0]) % sparse_mask.shape[
            0
        ] == 0, f"sparse_mask must divide the query dimension, got {sparse_mask.shape[0]} and {sparse_range_query}"
        assert (sparse_range_key_value[1] - sparse_range_key_value[0]) % sparse_mask.shape[
            1
        ] == 0, (
            f"sparse_mask must divide the key/value dimension, got {sparse_mask.shape[1]} and {sparse_range_key_value}"
        )

        query_left, query_mid, query_right = query.split(
            [
                sparse_range_query[0],
                sparse_range_query[1] - sparse_range_query[0],
                s_q - sparse_range_query[1],
            ],
            dim=2,
        )
        key_left, key_mid, key_right = key.split(
            [
                sparse_range_key_value[0],
                sparse_range_key_value[1] - sparse_range_key_value[0],
                s_kv - sparse_range_key_value[1],
            ],
            dim=2,
        )
        value_left, value_mid, value_right = value.split(
            [
                sparse_range_key_value[0],
                sparse_range_key_value[1] - sparse_range_key_value[0],
                s_kv - sparse_range_key_value[1],
            ],
            dim=2,
        )

        output = []

        if query_left.numel() > 0:
            output.append(
                F.scaled_dot_product_attention(
                    query_left,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            )

        if query_mid.numel() > 0:
            sdpa_merger = torch_ring_attention._SDPAMerger(
                not para_attn.config.attention.allow_reduced_precision_compute
            )

            if key_left.numel() > 0:
                sdpa_merger.step(
                    *para_attn_ops.attention_forward_with_lse(
                        query_mid,
                        key_left,
                        value_left,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                    )
                )

            if key_mid.numel() > 0:
                sparse_output, sparse_lse = [], []
                query_chunk_size = query_mid.shape[2] // sparse_mask.shape[0]
                key_value_chunk_size = key_mid.shape[2] // sparse_mask.shape[1]
                if print_attn_weight_means:
                    attn_weight = torch.einsum("bhqd,bhkd->bhqk", query_mid, key_mid) * (
                        d_qk**-0.5 if scale is None else scale
                    )
                    attn_weight_means = (
                        F.softmax(attn_weight, dim=-1)
                        .unflatten(3, (-1, key_value_chunk_size))
                        .unflatten(2, (-1, query_chunk_size))
                        .sum(dim=-1)
                        .mean(dim=(0, 1, 3))
                    )
                    print(attn_weight_means)
                    del attn_weight
                    del attn_weight_means
                for i, (mask_row, query_chunk) in enumerate(
                    zip(sparse_mask, query_mid.chunk(sparse_mask.shape[0], dim=2))
                ):
                    sub_sdpa_merger = torch_ring_attention._SDPAMerger(
                        not para_attn.config.attention.allow_reduced_precision_compute
                    )
                    start = 0
                    while start < mask_row.shape[0]:
                        if not mask_row[start]:
                            start += 1
                            continue
                        end = start + 1
                        while end < mask_row.shape[0] and mask_row[end]:
                            end += 1
                        sub_sdpa_merger.step(
                            *para_attn_ops.attention_forward_with_lse(
                                query_chunk,
                                key_mid[:, :, start * key_value_chunk_size : end * key_value_chunk_size],
                                value_mid[:, :, start * key_value_chunk_size : end * key_value_chunk_size],
                                attn_mask=attn_mask,
                                dropout_p=dropout_p,
                                is_causal=is_causal,
                                scale=scale,
                            )
                        )
                        start = end + 1
                    row_output, row_lse = sub_sdpa_merger.results()
                    sparse_output.append(row_output)
                    sparse_lse.append(row_lse)
                    del sub_sdpa_merger
                    del row_output
                    del row_lse
            sparse_output = torch.cat(sparse_output, dim=2)
            sparse_lse = torch.cat(sparse_lse, dim=2)
            sdpa_merger.step(sparse_output, sparse_lse)
            del sparse_output
            del sparse_lse

            if key_right.numel() > 0:
                sdpa_merger.step(
                    *para_attn_ops.attention_forward_with_lse(
                        query_mid,
                        key_right,
                        value_right,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                    )
                )

            output.append(sdpa_merger.results()[0])
            del sdpa_merger

        if query_right.numel() > 0:
            output.append(
                F.scaled_dot_product_attention(
                    query_right,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            )

        output = torch.cat(output, dim=2)
        output = output.to(query.dtype)
        return output

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError("Backward pass for StructuredSparseAttnFunc is not implemented")


def struct_sparse_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    sparse_mask=None,
    sparse_range_query=None,
    sparse_range_key_value=None,
    print_attn_weight_means=False,
):
    return StructuredSparseAttnFunc.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        sparse_mask,
        sparse_range_query,
        sparse_range_key_value,
        print_attn_weight_means,
    )


class FocusAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        downsample_factor,
        focus_mask,
        focus_range_query,
        focus_range_key_value,
        print_attn_weight_means,
    ):
        if focus_mask is None:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

        assert torch_ring_attention is not None, "FocusAttnFunc requires a newer version of PyTorch"

        assert query.ndim == 4, "query must have 4 dimensions, got {}".format(query.ndim)
        assert key.ndim == 4, "key must have 4 dimensions, got {}".format(key.ndim)
        assert value.ndim == 4, "value must have 4 dimensions, got {}".format(value.ndim)

        assert not is_causal, "is_causal is not supported in FocusAttnFunc"

        assert focus_mask.ndim == 2, "focus_mask must have 2 dimensions, got {}".format(focus_mask.ndim)

        b, h, s_q, d_qk = query.shape
        _, _, s_kv, d_v = value.shape

        if focus_range_query is None:
            focus_range_query = (0, s_q)
        if focus_range_key_value is None:
            focus_range_key_value = (0, s_kv)

        assert (
            0 <= focus_range_query[0] <= focus_range_query[1] <= s_q
        ), f"focus_range_query must be in [0, {s_q}], got {focus_range_query}"
        assert (
            0 <= focus_range_key_value[0] <= focus_range_key_value[1] <= s_kv
        ), f"focus_range_key_value must be in [0, {s_kv}], got {focus_range_key_value}"

        assert (focus_range_query[1] - focus_range_query[0]) % focus_mask.shape[
            0
        ] == 0, f"focus_mask must divide the query dimension, got {focus_mask.shape[0]} and {focus_range_query}"
        assert (focus_range_key_value[1] - focus_range_key_value[0]) % focus_mask.shape[
            1
        ] == 0, f"focus_mask must divide the key/value dimension, got {focus_mask.shape[1]} and {focus_range_key_value}"

        query_left, query_mid, query_right = query.split(
            [
                focus_range_query[0],
                focus_range_query[1] - focus_range_query[0],
                s_q - focus_range_query[1],
            ],
            dim=2,
        )
        key_left, key_mid, key_right = key.split(
            [
                focus_range_key_value[0],
                focus_range_key_value[1] - focus_range_key_value[0],
                s_kv - focus_range_key_value[1],
            ],
            dim=2,
        )
        value_left, value_mid, value_right = value.split(
            [
                focus_range_key_value[0],
                focus_range_key_value[1] - focus_range_key_value[0],
                s_kv - focus_range_key_value[1],
            ],
            dim=2,
        )

        output = []

        if query_left.numel() > 0:
            output.append(
                F.scaled_dot_product_attention(
                    query_left,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            )

        if query_mid.numel() > 0:
            sdpa_merger = torch_ring_attention._SDPAMerger(
                not para_attn.config.attention.allow_reduced_precision_compute
            )

            if key_left.numel() > 0:
                sdpa_merger.step(
                    *para_attn_ops.attention_forward_with_lse(
                        query_mid,
                        key_left,
                        value_left,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                    )
                )

            if key_mid.numel() > 0:
                if downsample_factor == 2:
                    # key_downsampled = (key_mid[:, :, ::2] + key_mid[:, :, 1::2]) * 0.5
                    # value_downsampled = (value_mid[:, :, ::2] + value_mid[:, :, 1::2]) * 0.5
                    # key_downsampled = key_mid[:, :, ::2]
                    # value_downsampled = value_mid[:, :, ::2]
                    indices = torch.randint(0, 2, (key_mid.shape[2] // 2,), device=key_mid.device) + torch.arange(
                        0, key_mid.shape[2], 2, device=key_mid.device
                    )
                    key_downsampled = key_mid.index_select(2, indices)
                    value_downsampled = value_mid.index_select(2, indices)
                else:
                    raise NotImplementedError(f"downsample_factor={downsample_factor} is not supported")

                focus_output, focus_lse = [], []
                query_chunk_size = query_mid.shape[2] // focus_mask.shape[0]
                key_value_chunk_size = key_mid.shape[2] // focus_mask.shape[1]
                key_value_chunk_size_downsampled = key_downsampled.shape[2] // focus_mask.shape[1]
                if print_attn_weight_means:
                    attn_weight = torch.einsum("bhqd,bhkd->bhqk", query_mid, key_mid) * (
                        d_qk**-0.5 if scale is None else scale
                    )
                    attn_weight_means = (
                        F.softmax(attn_weight, dim=-1)
                        .unflatten(3, (-1, key_value_chunk_size))
                        .unflatten(2, (-1, query_chunk_size))
                        .sum(dim=-1)
                        .mean(dim=(0, 1, 3))
                    )
                    print(attn_weight_means)
                    del attn_weight
                    del attn_weight_means
                for i, (mask_row, query_chunk) in enumerate(
                    zip(focus_mask, query_mid.chunk(focus_mask.shape[0], dim=2))
                ):
                    sub_sdpa_merger = torch_ring_attention._SDPAMerger(
                        not para_attn.config.attention.allow_reduced_precision_compute
                    )
                    start = 0
                    while start < mask_row.shape[0]:
                        end = start + 1
                        while end < mask_row.shape[0] and mask_row[start] == mask_row[end]:
                            end += 1
                        if mask_row[start]:
                            sub_sdpa_merger.step(
                                *para_attn_ops.attention_forward_with_lse(
                                    query_chunk,
                                    key_mid[:, :, start * key_value_chunk_size : end * key_value_chunk_size],
                                    value_mid[:, :, start * key_value_chunk_size : end * key_value_chunk_size],
                                    attn_mask=attn_mask,
                                    dropout_p=dropout_p,
                                    is_causal=is_causal,
                                    scale=scale,
                                )
                            )
                        else:
                            # sub_sdpa_merger.step(
                            #     *para_attn_ops.attention_forward_with_lse(
                            #         query_chunk,
                            #         key_downsampled[:, :, start * key_value_chunk_size_downsampled : end * key_value_chunk_size_downsampled],
                            #         value_downsampled[:, :, start * key_value_chunk_size_downsampled : end * key_value_chunk_size_downsampled],
                            #         attn_mask=attn_mask,
                            #         dropout_p=dropout_p,
                            #         is_causal=is_causal,
                            #         scale=scale,
                            #     )
                            # )
                            out_downsampled, lse_downsampled = para_attn_ops.attention_forward_with_lse(
                                query_chunk,
                                key_downsampled[
                                    :,
                                    :,
                                    start * key_value_chunk_size_downsampled : end * key_value_chunk_size_downsampled,
                                ],
                                value_downsampled[
                                    :,
                                    :,
                                    start * key_value_chunk_size_downsampled : end * key_value_chunk_size_downsampled,
                                ],
                                attn_mask=attn_mask,
                                dropout_p=dropout_p,
                                is_causal=is_causal,
                                scale=scale,
                            )
                            if downsample_factor == 2:
                                lse_downsampled = lse_downsampled - (-0.69314718)
                            else:
                                raise NotImplementedError(f"downsample_factor={downsample_factor} is not supported")
                            sub_sdpa_merger.step(out_downsampled, lse_downsampled)
                            del out_downsampled
                            del lse_downsampled
                        start = end + 1
                    row_output, row_lse = sub_sdpa_merger.results()
                    focus_output.append(row_output)
                    focus_lse.append(row_lse)
                    del sub_sdpa_merger
                    del row_output
                    del row_lse
                del key_downsampled
                del value_downsampled
            focus_output = torch.cat(focus_output, dim=2)
            focus_lse = torch.cat(focus_lse, dim=2)
            sdpa_merger.step(focus_output, focus_lse)
            del focus_output
            del focus_lse

            if key_right.numel() > 0:
                sdpa_merger.step(
                    *para_attn_ops.attention_forward_with_lse(
                        query_mid,
                        key_right,
                        value_right,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                    )
                )

            output.append(sdpa_merger.results()[0])
            del sdpa_merger

        if query_right.numel() > 0:
            output.append(
                F.scaled_dot_product_attention(
                    query_right,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            )

        output = torch.cat(output, dim=2)
        output = output.to(query.dtype)
        return output

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError("Backward pass for StructuredSparseAttnFunc is not implemented")


def focus_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    downsample_factor=2,
    focus_mask=None,
    focus_range_query=None,
    focus_range_key_value=None,
    print_attn_weight_means=False,
):
    return FocusAttnFunc.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        downsample_factor,
        focus_mask,
        focus_range_query,
        focus_range_key_value,
        print_attn_weight_means,
    )


class SparseKVAttnMode(BaseTorchFunctionMode):
    disabled = False

    @torch.compiler.disable
    def __init__(self, *, dispatch_to_custom_ops=True):
        super().__init__()
        self._dispatch_to_custom_ops = dispatch_to_custom_ops

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        print(f"SparseKVAttnMode used")

        if SparseKVAttnMode.disabled:
            return base_handle_torch_function(func, types, args, kwargs)

        if func is F.scaled_dot_product_attention:
            return sparse_kv_attn_func(*args, **kwargs, dispatch_to_custom_ops=self._dispatch_to_custom_ops)

        return base_handle_torch_function(func, types, args, kwargs)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class StructSparseAttnMode(BaseTorchFunctionMode):
    disabled = False

    @torch.compiler.disable
    def __init__(
        self,
        *,
        sparse_mask=None,
        sparse_range_query=None,
        sparse_range_key_value=None,
        print_attn_weight_means=False,
    ):
        super().__init__()
        self._sparse_mask = sparse_mask
        self._sparse_range_query = sparse_range_query
        self._sparse_range_key_value = sparse_range_key_value
        self._print_attn_weight_means = print_attn_weight_means

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if StructSparseAttnMode.disabled:
            return base_handle_torch_function(func, types, args, kwargs)

        if func is F.scaled_dot_product_attention:
            return struct_sparse_attn_func(
                *args,
                **kwargs,
                sparse_mask=self._sparse_mask,
                sparse_range_query=self._sparse_range_query,
                sparse_range_key_value=self._sparse_range_key_value,
                print_attn_weight_means=self._print_attn_weight_means,
            )

        return base_handle_torch_function(func, types, args, kwargs)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class FocusAttnMode(BaseTorchFunctionMode):
    disabled = False

    @torch.compiler.disable
    def __init__(
        self,
        *,
        downsample_factor=2,
        focus_mask=None,
        focus_range_query=None,
        focus_range_key_value=None,
        print_attn_weight_means=False,
    ):
        super().__init__()
        self._downsample_factor = downsample_factor
        self._focus_mask = focus_mask
        self._focus_range_query = focus_range_query
        self._focus_range_key_value = focus_range_key_value
        self._print_attn_weight_means = print_attn_weight_means

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if FocusAttnMode.disabled:
            return base_handle_torch_function(func, types, args, kwargs)

        if func is F.scaled_dot_product_attention:
            return focus_attn_func(
                *args,
                **kwargs,
                downsample_factor=self._downsample_factor,
                focus_mask=self._focus_mask,
                focus_range_query=self._focus_range_query,
                focus_range_key_value=self._focus_range_key_value,
                print_attn_weight_means=self._print_attn_weight_means,
            )

        return base_handle_torch_function(func, types, args, kwargs)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled
