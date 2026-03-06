import torch
import torch
import os   
import argparse
import yaml
import sys
import gc 
from utils import load_prompts
from functools import partial
from pipeline_qwenimage import QwenImagePipeline
from transformer_qwenimage import QwenImageTransformer2DModel,set_attn
import diffusers.models
original_transformer = diffusers.models.QwenImageTransformer2DModel
diffusers.models.QwenImageTransformer2DModel = QwenImageTransformer2DModel




def clear_cuda():
    torch.cuda.empty_cache()
    gc.collect()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Qwen Image Generation with Config')
    parser.add_argument('--config', type=str, default='', help='Path to config file (default: config.yaml)')
    args = parser.parse_args()
    
    config_path = args.config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
    except:
        print(f"Config file not found: {config_path}")
        sys.exit(1)
          
    height, width = config['base']['height'], config['base']['width']
    method_name_height = config['method']['name_height']
    method_name_width = config['method']['name_width']
    seed = config['base']['seed']
    method = config['method']
    method['hyper_parameters_height']['linear_factor'] = height/1328
    method['hyper_parameters_width']['linear_factor'] = width/1328
    guidance_scale = config['base']['guidance_scale']


    class MyTransformer(QwenImageTransformer2DModel):
        def __init__(self, *args, **kwargs):
            kwargs["method"] = method
            super().__init__(*args, **kwargs)

    diffusers.models.QwenImageTransformer2DModel = MyTransformer
    output_dir = os.path.join(config['base']['output_dir'],f"{height}x{width}", f"{method_name_height}_{method_name_width}")
    os.makedirs(output_dir, exist_ok=True)
    
    prompts =  load_prompts(config['base']['prompts_file'])
    pipe = QwenImagePipeline.from_pretrained(
        config['base']['model_path'],
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
    )
    set_attn(pipe.transformer, method=method)
    for fn in ["enable_model_cpu_offload", "enable_attention_slicing"]:
        try:
            getattr(pipe, fn)()
        except Exception:
            pass
    try:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
    except Exception:
        pass
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = torch.Generator(device=gen_device).manual_seed(seed)
    for i, prompt in enumerate(prompts):
        print("running prompt:",prompt)
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        save_path = f"{method.get('save_dir',output_dir)}/{safe_prompt}"
        method['save_path'] = save_path
        attn_func = method.get('attention_operator',{}).get('attn_func','standard')
        num_steps = method.get("num_steps",50)
        if attn_func == 'attention_decay' and not os.path.exists(save_path):
            print("Do not have block avg, start to generate for attention decay")
            os.makedirs(save_path,exist_ok=True)
            method["attention_operator"]["attn_func"] = "get_block_avg"
            method["num_steps"]= 2
            with torch.inference_mode():
                out = pipe(
                    prompt=prompt,
                    negative_prompt="",
                    width=width,
                    height=height,
                    num_inference_steps=method.get('num_steps',50),
                    true_cfg_scale=guidance_scale,
                    generator=generator,
                    method=method,               
                )
                del out
        method["attention_operator"]["attn_func"] = attn_func
        method["num_steps"]= num_steps
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                negative_prompt="repeated elements, warped geometry, distorted image",
                width=width,
                height=height,
                num_inference_steps=method.get('num_steps',50),
                true_cfg_scale=guidance_scale,
                generator=generator,
                method=method,               
            )
        image = out.images[0]
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        filename = f"{output_dir}/{i+1:03d}_{safe_prompt}.png"
        image.save(filename)
        print(f"Saved: {filename}")
    
    del pipe
    clear_cuda()
    diffusers.models.QwenImageTransformer2DModel = original_transformer
    print("✅ All done!")
