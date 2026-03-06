import os
import shutil
import datetime
import yaml
import torch
import numpy as np


def load_prompts(file_path):
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  
                prompts.append(line)
    prompts = prompts
    return prompts


def pad_qkv(input_tensor, block_size=128):
    """
    Pad the input tensor to be a multiple of the block size.
    input shape: (seqlen, num_heads, hidden_dim)
    """
    bsz, num_heads,seqlen, hidden_dim = input_tensor.shape
    # Calculate the necessary padding
    padding_length = (block_size - (seqlen % block_size)) % block_size
    # Create a padded tensor with zeros
    padded_tensor = torch.zeros((bsz,num_heads,seqlen + padding_length,  hidden_dim), device=input_tensor.device, dtype=input_tensor.dtype)
    # Copy the original tensor into the padded tensor
    padded_tensor[:,:,:seqlen, :] = input_tensor
    
    return padded_tensor


def get_continuous_decay(
    block_entropy,
    h_min = 0.04,
    h_max = 6.25,
    lambda_max = 1.3,
    lambda_min = 1.2,
    func_type = "linear",
    gamma = 1.0,
):  
    b,h,block_num = block_entropy.shape
    entropy = block_entropy.mean(dim=-1)
    decay_mask = torch.ones((b,h,block_num,block_num),dtype=block_entropy.dtype,device=block_entropy.device)  
    if func_type == "linear":
        decay_factor = linear_decay(entropy, alpha=lambda_max, beta=lambda_min, x_min=h_min, x_max=h_max)
    
    elif func_type == "power":
        decay_factor = power_decay(entropy,alpha=lambda_max,beta=lambda_min, gamma= gamma,x_min=h_min,x_max=h_max)
    for i in range(b):
        for j in range(h):
            decay_mask[i,j] *= (decay_factor[i,j])
    
    return decay_mask

def linear_decay(entropy, alpha=1.0, beta=0.1, x_min=0.04, x_max=6.24):
    entropy = torch.clip(entropy,x_min,x_max)
    m = (beta - alpha) / (x_max - x_min)
    c = alpha - m * x_min
    
    decay_factor = m * entropy + c
    return torch.clip(decay_factor, beta, alpha)


def normalize_entropy(x, x_min=0.04, x_max=6.25):
    """Normalize entropy x to [0, 1]"""
    return (x - x_min) / (x_max - x_min)

def map_to_range(g_z, alpha=1.3, beta=1.0):
    """Map the base decay function g(z) from [1, 0] to [alpha, beta]"""
    return beta + (alpha - beta) * g_z

def power_decay(x, alpha=1.3,beta=1.0,gamma=1.0,x_min=0.04,x_max=6.25):
    z = normalize_entropy(x,x_min=x_min,x_max=x_max)
    g_z = torch.pow(1.0 - z, gamma)
    return map_to_range(g_z,alpha=alpha,beta=beta)
