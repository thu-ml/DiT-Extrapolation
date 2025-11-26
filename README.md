## Diffusion-transformer Extrapolation for Long/High-resolution Video/Image Generation
This repository provides the official implementation of [RIFLEx](https://arxiv.org/abs/2502.15894) and [UltraViCo](https://arxiv.org/abs/2511.20123), which achieve diffusion-transformer extrapolation for long video generation in a plug-and-play way.

<div align="center">
<img src='assets/riflex.png'></img>

<a href="https://huggingface.co/papers/2502.15894"><img src="https://img.shields.io/static/v1?label=Daily papers&message=HuggingFace&color=yellow"></a>
<a href='https://riflex-video.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 
<a href='https://arxiv.org/pdf/2502.15894'><img src='https://img.shields.io/badge/arXiv-2502.15894-b31b1b.svg'></a> &nbsp;
<a href='https://www.youtube.com/watch?v=taofoXDsKGk'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a><br>



<div>
    <a href="https://gracezhao1997.github.io/" target="_blank">Min Zhao</a><sup></sup> | 
    <a href="https://guandehe.github.io/" target="_blank">Guande He</a><sup></sup> | 
    <a href="https://github.com/Chyxx" target="_blank">Yixiao Chen</a><sup></sup> | 
    <a href="https://zhuhz22.github.io/" target="_blank">Hongzhou Zhu</a><sup></sup>|
<a href="https://zhenxuan00.github.io/" target="_blank">Chongxuan Li</a><sup></sup> | 
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml" target="_blank">Jun Zhu</a><sup></sup>
</div>
<div>
    <sup></sup>Tsinghua University
</div>



</div>

---

## 🎉 Supported Models
Here, we list the SOTA video diffusion transformers that RIFLEx has been applied to. We are continuously working to support more models. Feel free to suggest additional models you would like us to support!


| Model                                                   | Extrapolation | Example Results                                              |  
|---------------------------------------------------------|---------------|--------------------------------------------------------------|  
| [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) | 5s -> 11s     | <img src=assets/example/hun-free-RIFLEx.gif width="250">     | 
| [CogVideoX-5B](https://github.com/THUDM/CogVideo)       | 6s -> 12s     | <img src=assets/example/cog-finetune-RIFLEx.gif width="250"> |
| [Wan2.1](https://github.com/Wan-Video/Wan2.1)             | 5s -> 8s     | <img src=assets/example/wan-free.gif width="250">            |

To be continued…… 

## 🔥🔥 News
- **2025.5.1** : RIFLEx is accepted by ICML 2025!
- **2025.3.17** : RIFLEx is integrated to the official [HunyuanVideo-I2V](https://github.com/Tencent/HunyuanVideo-I2V) repository!
- **2025.3.1** : The code for [CogVideoX-5B](https://github.com/THUDM/CogVideo) and fine-tuned [CogVideoX-RIFLEx](https://huggingface.co/thu-ml/CogVideoX-RIFLEx-diffusers/tree/main) are released.
- **2025.2.26** RIFLEx is supported in [HunyuanVideoGP](https://github.com/deepbeepmeep/HunyuanVideoGP), where a 10.5 s video at 1280x720 can be generated on an RTX 4090.
- **2025.2.26** RIFLEx is supported in [ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper) by [KiJai](https://github.com/kijai).
- **2025.2.25** : The [paper](https://arxiv.org/pdf/2502.15894), [project page](https://riflex-video.github.io/), code and fine-tuned [HunyuanVideo-RIFLEx](https://huggingface.co/thu-ml/Hunyuan-RIFLEx-diffusers) are released.


## RIFLEx Code
RIFLEx only adds a single line of code on the original [1D RoPE](https://github.com/huggingface/diffusers/blob/9c7e205176c30b27c5f44ec7650a8dfcc12dde86/src/diffusers/models/embeddings.py#L1105).
```python
def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    '''
        k: the index for the intrinsic frequency in RoPE
        L_test: the number of frames for inference
    '''
    
    assert dim % 2 == 0
    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim)) 

    # === RIFLEx modification start ===
    # Reduce intrinsic frequency to stay within a single period after extrapolation (Eq.(8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply 0.9 to keep extrapolated length below 90% of a period. 
    freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === RIFLEx modification end ===

    freqs = torch.outer(pos, freqs)  
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  
    return freqs_cos, freqs_sin
```
In `riflex_utils.py`, we show how to identify the intrinsic frequency in a RoPE-based pre-trained diffusion transformer.

## Single GPU Inference with Diffusers for Quick Start
### Installation
```bash
conda create -n riflex python=3.10
pip install -r requirements.txt
pip install -U bitsandbytes
```

### Prompts
The example prompts for all models are listed in `assets/prompts`. The prompts on the project page can be found in `assets/prompts/free_hunyuan.txt` and `assets/prompts/finetune_hunyuan.txt`. 

Please note that for single GPU inference with HunyuanVideo, Diffusers use `DiffusersBitsAndBytesConfig` to save memory, which may affect performance. To produce the demo on the [project page](https://riflex-video.github.io/), please refer to the [Multi-GPU Inference](#multi-gpu-inference--recommended-) section.

### Inference for HunyuanVideo

<details>
<summary> 2× temporal extrapolation (click to expand)</summary>

For training-free: 
```bash
python hunyuanvideo.py --k 4 --N_k 50 --num_frames 261 --prompt "A white and orange tabby cat is seen happily darting through a dense garden, as if chasing something. Its eyes are wide and happy as it jogs forward, scanning the branches, flowers, and leaves as it walks. The path is narrow as it makes its way between all the plants. the scene is captured from a ground-level angle, following the cat closely, giving a low and intimate perspective. The image is cinematic with warm tones and a grainy texture. The scattered daylight between the leaves and plants above creates a warm contrast, accentuating the cat’s orange fur. The shot is clear and sharp, with a shallow depth of field."
```

For fine-tuned [HunyuanVideo-RIFLEx](https://huggingface.co/thu-ml/Hunyuan-RIFLEx-diffusers): 

```bash
python hunyuanvideo.py --k 4 --N_k 66 --num_frames 261 --finetune --model_id "thu-ml/Hunyuan-RIFLEx-diffusers" --prompt "3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest."
```
> Note that the current version of diffusers only supports single-GPU inference. If there are multiple GPUs in the environment, please specify one by exporting CUDA_VISIBLE_DEVICES.
</details>

### Inference for CogVideoX

<details>

<summary> 2× temporal extrapolation (click to expand)</summary>

For training-free: 
```bash
python cogvideox.py --k 2 --N_k 20 --num_frames 97 --prompt "3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest."
```

For fine-tuned [CogVideoX-RIFLEx](https://huggingface.co/thu-ml/CogVideoX-RIFLEx-diffusers/tree/main): 
```bash
python cogvideox.py --k 2 --N_k 25 --num_frames 97 --finetune --model_id "thu-ml/CogVideoX-RIFLEx-diffusers" --prompt "A drone camera circles around a beautiful historic church built on a rocky outcropping along the Amalfi Coast, the view showcases historic and magnificent architectural details and tiered pathways and patios, waves are seen crashing against the rocks below as the view overlooks the horizon of the coastal waters and hilly landscapes of the Amalfi Coast Italy, several distant people are seen walking and enjoying vistas on patios of the dramatic ocean views, the warm glow of the afternoon sun creates a magical and romantic feeling to the scene, the view is stunning captured with beautiful photography."
```
</details>

## Multi-GPU Inference ( *Recommended* )
To **enhance inference speed** and **reproduce the demos** in our [project page](https://riflex-video.github.io/), please use the multi-gpu inference. Details can be found in the [`multi-gpu` branch](https://github.com/thu-ml/RIFLEx/tree/multi-gpu).

## References
If you find the code useful, please cite
```
@article{zhao2025riflex,
  title={Riflex: A free lunch for length extrapolation in video diffusion transformers},
  author={Zhao, Min and He, Guande and Chen, Yixiao and Zhu, Hongzhou and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2502.15894},
  year={2025}
}
```
