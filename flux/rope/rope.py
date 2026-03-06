import torch
import torch.nn as nn
from typing import List

import torch
import torch.nn as nn
from typing import List
from typing import Dict, Any

def forward_rope(self, ids: torch.Tensor, ntk_factor: float, method: Dict[str, Any]) -> torch.Tensor:
    n_axes = ids.shape[-1]
    # text
    emb_0 = extrapolation(ids[..., 0] / 1, self.axes_dim[0], self.theta)

    # height 
    height_method_name = method.get('name_height', 'extrapolation')
    height_hyper = method.get('hyper_parameters_height', {})
    height_rope_func = globals()[height_method_name]
    emb_1 = height_rope_func(
        ids[..., 1] / 1,
        self.axes_dim[1],
        self.theta,
        height_hyper,
    )

    # width
    width_method_name = method.get('name_width', 'extrapolation')
    width_hyper = method.get('hyper_parameters_width', {})
    width_rope_func = globals()[width_method_name]
    emb_2 = width_rope_func(
            ids[..., 2] / 1,
            self.axes_dim[2],
            self.theta,
            width_hyper,
        )

    # concat
    emb = torch.cat([emb_0, emb_1, emb_2], dim=-3) # [1, 1, seq, 64, 2, 2]
    return emb.unsqueeze(1)

def extrapolation(pos: torch.Tensor, dim: int, theta: int, hyper: List[float]=None) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    
    if hyper is None:
        ntk_factor = 1
    else:
        ntk_factor = hyper.get('ntk_factor', 1)

    batch_size, seq_length = pos.shape
    
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk_factor)**scale)

    '''if ntk_factor > 1:
        import math
        omega_inter = 1.0 / (theta**scale) / math.sqrt((seq_length - 512) / 64 ** 2)
        omega = torch.max(omega, omega_inter)'''

    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()

def ntk(pos: torch.Tensor, dim: int, theta: int, hyper: List[float]=None) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    
    batch_size, seq_length = pos.shape
    train_length = hyper.get('train_length', 1024)
        
    if seq_length <= train_length:
        ntk_factor = 1.0
    else:
        ntk_factor = hyper.get('ntk_factor', 1)
    
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk_factor)**scale)

    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()

def multi_inter(pos: torch.Tensor, dim: int, theta: int, hyper: List[float]=None):
    batch_size, seq_length = pos.shape

    freq_ids = hyper.get('freq_id', [])  #here it is a list
    train_length = hyper.get('train_length', 1024)

    if seq_length <= train_length:
        linear_factor = 1.0
    else:
        linear_factor = hyper.get('linear_factor', 1.0)

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    freqs = 1.0 / (theta ** scale)

    if linear_factor > 1.0:
        freqs_inter = freqs / linear_factor
        freqs[freq_ids] = freqs_inter[freq_ids]  #replace the freqs_extra with freqs_inter

    out = torch.einsum("...n,d->...nd", pos, freqs)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()

def interpolation(pos: torch.Tensor, dim: int, theta: int, hyper: List[float]=None):
    batch_size, seq_length = pos.shape

    train_length = hyper.get('train_length', 1024)

    if seq_length <= train_length:
        linear_factor = 1.0
    else:
        linear_factor = hyper.get('linear_factor', 1.0)

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    freqs = 1.0 / (theta ** scale) / linear_factor
    out = torch.einsum("...n,d->...nd", pos, freqs)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()



def yarn(pos: torch.Tensor, dim: int, theta: int, hyper: List[float]=None):
    batch_size, seq_length = pos.shape
    train_length = hyper.get('train_length', 1024)

    if seq_length <= train_length:
        linear_factor = 1.0
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / ((theta)**scale)
        out = torch.einsum("...n,d->...nd", pos, omega)
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    else:
        linear_factor = hyper.get('linear_factor', 1.0)
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        freqs = 1.0 / (theta ** scale)
        
        interpolation_mask = (1/linear_factor)*torch.ones(freqs.shape[0], dtype=freqs.dtype, device=freqs.device)
        alpha=1
        beta=32
        r = ((1328)*freqs)/(2*torch.pi)
        alpha_t = r.new_tensor(alpha)
        beta_t  = r.new_tensor(beta)
        gamma = ((r - alpha_t) / (beta_t - alpha_t)).clamp(0.0, 1.0)
        freqs=gamma*freqs + (1-gamma)*freqs*interpolation_mask
        out = torch.einsum("...n,d->...nd", pos, freqs)
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()

def riflex(pos: torch.Tensor, dim: int, theta: int, hyper: List[float]=None) -> torch.Tensor:
    batch_size, seq_length = pos.shape
    train_length = hyper.get('train_length', 1024)
    if seq_length <= train_length:
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / ((theta)**scale)
        out = torch.einsum("...n,d->...nd", pos, omega)
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    else:
        freq_ids = hyper.get('freq_id', [])
        length = hyper.get('length', 200)
        riflex_scale = hyper.get('riflex_scale', 0.9)
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        freqs_extra = 1.0 / ((theta)**scale)
        theta_i = riflex_scale * 2 * torch.pi / length
        freqs_extra[freq_ids]=theta_i
        out = torch.einsum("...n,d->...nd", pos, freqs_extra)
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()