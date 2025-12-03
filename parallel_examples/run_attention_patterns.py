import torch
import torch.distributed as dist
from diffusers import HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers.utils import export_to_video
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from para_attn.sage_attn.core import sage_attention
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae


def init_distributed():
    dist.init_process_group()
    torch.cuda.set_device(dist.get_rank())

def load_model():
    model_id = "hunyuanvideo-community/HunyuanVideo"
    
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    
    pipe = HunyuanVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.float16,
    ).to("cuda")
    
    return pipe



def parallelize_model(pipe, mesh, attention_args):
    parallelize_pipe(
        pipe,
        mesh=mesh,
        new_attention=sage_attention,
        attention_args=attention_args
    )
    parallelize_vae(pipe.vae, mesh=mesh._flatten())

def generate_video(pipe, prompt, height, width, num_frames, num_inference_steps, generator_seed):
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        output_type="pil" if dist.get_rank() == 0 else "pt",
        generator=torch.Generator(device="cuda").manual_seed(generator_seed),
    ).frames[0]
    
    return output

def save_video(output, output_dir, filename):
    if dist.get_rank() == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        export_to_video(output, os.path.join(output_dir, filename), fps=24)
        print(f"Video saved to: {os.path.join(output_dir, filename)}")
def cleanup_model_state(pipe):
    try:
        if hasattr(pipe.vae, 'disable_tiling'):
            pipe.vae.disable_tiling()
        if hasattr(pipe.transformer, 'attn_processors'):
            from diffusers.models.attention_processor import AttnProcessor2_0
            default_processors = {}
            for name in pipe.transformer.attn_processors.keys():
                default_processors[name] = AttnProcessor2_0()
            pipe.transformer.set_attn_processor(default_processors)

        if hasattr(pipe, 'to'):
            pipe.to('cpu')
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def force_memory_cleanup():
    import gc
    
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def cleanup_distributed_state():
    try:
        if dist.is_initialized():
            dist.barrier()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
    except Exception as e:
        print(f"Warning: Error during distributed cleanup: {e}")

def run_experiment(config):

    prompt_name = '_'.join(config['prompt'].split(' ')[0:15])

    pipe = load_model()
    
    num_frames = config['num_frames']
    height = config['height']
    width = config['width']
    prompt = config['prompt']
    num_inference_steps = config['num_inference_steps']
    generator_seed = config['generator_seed']
    
    attention_args = {
        'frame_tokens': 2040,
        'prompt_name': prompt_name,
        'num_frames': num_frames,
        'block_size': 128,
        'multi_factor': config['multi_factor'],
        'mask_factor': config['mask_factor'],
        'latents': config['latents'],
    }
            
    mesh = init_context_parallel_mesh(pipe.device.type)
    parallelize_model(
        pipe, mesh, attention_args
    )
    pipe.vae.enable_tiling()
    
    output = generate_video(pipe, prompt, height, width, num_frames, num_inference_steps, generator_seed)
    
    output_dir = f"output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    filename = f"{prompt_name}.mp4"
    # The original implementation of ParaAttention causes issues in the last few frames, even at the original training length, which is not introduced by our extrapolation algorithm. 
    # For now, we simply discard these frames and will properly fix this in future.
    output = output[:-8]
    save_video(output, output_dir, filename)

    del output
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return True
        
def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.6)
    parser.add_argument('--extrapolation_ratio', type=int, default=3)
    parser.add_argument('--height', type=int, default=544)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    args = parser.parse_args()


    num_frames = (33 * args.extrapolation_ratio - 1) * 4 + 1
    L_test = 33 * args.extrapolation_ratio
    config = {
        'height': args.height,
        'width': args.width,
        'num_inference_steps':args.num_inference_steps,
        'generator_seed': 42,
        'context_length': 256,
        'prompt': args.prompt,
        'num_frames': num_frames,
        'window_width': 33, 
        'multi_factor': args.alpha,
        'mask_factor': args.beta,
        'latents': L_test,
    }
    
    init_distributed()
    run_experiment(config)

           
            

if __name__ == "__main__":
    main() 