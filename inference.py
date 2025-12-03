from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
import torch, os
import os
import argparse
from diffusers.utils import export_to_video
from modify_wan import set_sage_attn_wan
from tqdm import tqdm
from sageattn.core import sage_attention

ATTNENTION = {
    "sage": sage_attention,
}

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--extrapolation_ratio', type=int, default=3)
parser.add_argument('--num_inference_steps', type=int, default=50)
args = parser.parse_args()

negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
torch.manual_seed(42)
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

num_frames = int((21 * args.extrapolation_ratio - 1) * 4 + 1)
latents = int(21 * args.extrapolation_ratio)

block_num = ((latents + 1) * 1560 + 127) // 128
prompt_name = '_'.join(args.prompt.split(' ')[0:15])
output_dir = f"output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
filename = f'{prompt_name}.mp4'
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16).to("cuda")

prompt_name = '_'.join(args.prompt.split(' ')[0:15])


set_sage_attn_wan(
    pipe.transformer,
    sage_attention,
    attention_args={
        'prompt_name': prompt_name,
        'num_frames': num_frames,
        'block_size': 128,
        'frame_tokens': 1560,
        'window_width': 21
    },
    prompt_name=prompt_name,
    block_size=128,
    multi_factor=args.alpha
    
)
video = pipe(
    prompt=args.prompt, 
    negative_prompt = negative_prompt,
    num_frames=num_frames,
    generator=torch.Generator(device="cuda").manual_seed(42),
    num_inference_steps=args.num_inference_steps
).frames[0]

export_to_video(video, f"{output_dir}/{filename}", fps=16)
