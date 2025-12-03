import torch
import torch.nn.functional as F
from typing import  List, Optional, Tuple, Union, Dict, Any
from diffusers.models.attention_processor import Attention
from diffusers.models import WanTransformer3DModel



class WanAttnProcessor2_0:
    def __init__(self,
                 attn_func,
                 attention_args=None,
                 
                 
                 block_id=None,
                 prompt_name=None,
                 
                 block_size=128,
                 
                 multi_factor=None,
            ):
        self.attn_func = attn_func
        self.attention_args = attention_args

        self.block_id = block_id
        self.cfg = False
        self.prompt_name = prompt_name

        self.block_size = block_size

        self.multi_factor = multi_factor
        
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        
        b, n, s, d = key.shape
        flags = (torch.zeros((b, n), dtype=torch.long)).to(key.device)
        hidden_states = self.attn_func(
            query, key, value, flags=flags, 
            is_causal=False, sm_scale=None, 
            text_false_length=0,multi_factor=self.multi_factor
        )
        
        
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
    
def set_sage_attn_wan(
        model: WanTransformer3DModel,
        attn_func,
        attention_args=None,
        prompt_name=None,
        block_size=128,
        multi_factor=None,
):

    original_forward = model.forward
    
    def forward_with_timestep(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not hasattr(self, 'timestep') or self.timestep != timestep:
            self.cfg = False
        else:
            self.cfg = True

        self.timestep = timestep

        for block in model.blocks:
            if hasattr(block, 'attn1') and hasattr(block.attn1, 'processor'):
                if hasattr(block.attn1.processor, 'timestep'):
                    block.attn1.processor.timestep = timestep
                    block.attn1.processor.cfg = self.cfg
        
        return original_forward(
            hidden_states, 
            timestep=timestep, 
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs
        )
    
    model.forward = forward_with_timestep.__get__(model) 
    for idx, block in enumerate(model.blocks):
        processor = WanAttnProcessor2_0(
            attn_func, 
            attention_args=attention_args,
            
            block_id=idx,
            prompt_name=prompt_name,
            
            block_size=block_size,
           
            multi_factor=multi_factor
        
        )
        block.attn1.processor = processor