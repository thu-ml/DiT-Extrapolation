import os  # noqa: C101
import sys

# import torch


_save_config_ignore = {
    # workaround: "Can't pickle <function ...>"
}


class attention:
    # https://github.com/pytorch/pytorch/blob/8dd380803c0e25786cba12801088c420a2ca071b/aten/src/ATen/native/transformers/attention.cpp#L574
    # https://github.com/pytorch/pytorch/blob/8dd380803c0e25786cba12801088c420a2ca071b/torch/_inductor/lowering.py#L2450
    force_dispatch_to_custom_ops = os.getenv("PARA_ATTN_FORCE_DISPATCH_TO_CUSTOM_OPS") == "1"

    allow_reduced_precision_compute = os.getenv("PARA_ATTN_ALLOW_REDUCED_PRECISION_COMPUTE") == "1"

    # fast_math = os.getenv("PARA_ATTN_FAST_MATH") == "1"


try:
    from torch.utils._config_module import install_config_module
except ImportError:
    # torch<2.2.0
    from torch._dynamo.config_utils import install_config_module

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
