import torch
from pipeline_flux_unified import FluxPipeline
from transformer_flux import FluxTransformer2DModel
import os
import argparse
import yaml
import sys
from rope.rope import forward_rope
from functools import partial
import shutil
import datetime
from utils import load_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='config for experiment')
    args = parser.parse_args()
    config_path = args.config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loading config: {config_path}")
    except Exception as e:
        print(f"❌ Fail to load config: {config_path}")
        print(f"   Error: {e}")
        sys.exit(1)
    
    base_config = config['base']
    model_path = base_config['model_path']
    height = base_config['resolution'][0]
    width = base_config['resolution'][1]
    seed = base_config['seed']
    prompts_low = base_config['prompts_low']
    prompts_high = base_config['prompts_high']
    train_height = base_config['train_resolution'][0]
    train_width = base_config['train_resolution'][1]
    native_resolution = train_height
   
    method_config = config['method']
    sdedit_threshold = method_config.get('sdedit_threshold', 20)
    text_duplication = method_config.get('text_duplication', True)
    proportional_attention = method_config.get('proportional_attention', True)
    ntk_factor = method_config.get('ntk_factor', 10)
    guidance_schedule = method_config.get('guidance_schedule', "cosine_decay")
    time_shift_2 = method_config.get('time_shift_2', 6)
    start_croords = method_config.get('start_croords', [0.5, 0.5])
    method_config['hyper_parameters_height']['linear_factor'] = height / 2048
    method_config['hyper_parameters_width']['linear_factor'] = width / 2048 
    method_config['hyper_parameters_height']['train_length'] = (train_height//16)*(train_width//16)+512
    method_config['hyper_parameters_width']['train_length'] = (train_height//16)*(train_width//16)+512
    
   
    output_dir = base_config['output_dir']
    image_dir = os.path.join(output_dir, "images")
    guidance_dir = os.path.join(output_dir, "guidance")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(guidance_dir, exist_ok=True)
    print(f"📁 Output dir: {output_dir}")
    
    prompts_low = load_prompts(prompts_low)
    prompts_high = load_prompts(prompts_high)
    print(f"📝 Loaded {len(prompts_low)} prompts.")
    
    print(f"🤖 Initialize...")
    transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = FluxPipeline.from_pretrained(model_path, transformer=None, torch_dtype=torch.bfloat16)
    pipe.transformer = transformer
    pipe.transformer.pos_embed.forward=partial(forward_rope, pipe.transformer.pos_embed, method=config['method'])
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.to("cuda")
    
    torch.random.manual_seed(seed)
    

    method=config['method']
    print(f"🎨 Begin generation...")
    for i, (prompt_low, prompt_high) in enumerate(zip(prompts_low, prompts_high)):
        safe_prompt = "".join(c for c in prompt_low[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        save_path = f"{method.get('save_dir',output_dir)}/{safe_prompt}"
        method['save_path'] = save_path
        attn_func = method.get('attention_operator',{}).get('attn_func','standard')
        if attn_func == 'attention_decay' and (not os.path.exists(save_path) or len(os.listdir(save_path))<57):
            print("Do not have block avg, start to generate for attention decay")
            os.makedirs(save_path,exist_ok=True)
            method["attention_operator"]["attn_func"] = "get_block_avg"
            method["num_steps"]= 2
            with torch.inference_mode():
                out = pipe(
                        prompt = prompt_low,
                        prompt_high = prompt_high,
                        generator=torch.Generator("cuda").manual_seed(0),
                        num_inference_steps1=2, num_inference_steps2=2, 
                        guidance_scale1=3.5, guidance_scale2=7,
                        height=height, width=width,
                        ntk_factor=ntk_factor,
                        return_dict=False,
                        time_shift_1=3, 
                        time_shift_2=time_shift_2,
                        proportional_attention = proportional_attention,
                        text_duplication = text_duplication,
                        swin_pachify = True,
                        guidance_schedule = guidance_schedule,
                        sdedit_threshold = sdedit_threshold,
                        native_resolution = native_resolution,
                        start_croords = start_croords,
                        hyper_params = method_config,
                        method=method,          
                )
                del out
        
            method["attention_operator"]["attn_func"] = attn_func
        images = pipe(
            prompt = prompt_low,
            prompt_high = prompt_high,
            generator=torch.Generator("cuda").manual_seed(0),
            num_inference_steps1=30, num_inference_steps2=30, 
            guidance_scale1=3.5, guidance_scale2=7,
            height=height, width=width,
            ntk_factor=ntk_factor,
            return_dict=False,
            time_shift_1=3, 
            time_shift_2=time_shift_2,
            proportional_attention = proportional_attention,
            text_duplication = text_duplication,
            swin_pachify = True,
            guidance_schedule = guidance_schedule,
            sdedit_threshold = sdedit_threshold,
            native_resolution = native_resolution,
            start_croords = start_croords,
            hyper_params = method_config,
            method=method,          
        )
        safe_prompt = "".join(c for c in prompt_low[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        filename = f"./{image_dir}/{i+1:03d}_{safe_prompt}.png"
        images[0].save(filename)
        guidance_filename = f"./{guidance_dir}/{i+1:03d}_{safe_prompt}.png"
        images[1].save(guidance_filename)
        print(f"Saved: {filename}")
        print(f"Saved: {guidance_filename}")