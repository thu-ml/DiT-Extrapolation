## Diffusion-Transformer Extrapolation for Long Video Generation
This repository provides the official implementation of [RIFLEx](https://arxiv.org/abs/2502.15894) and [UltraViCo](https://arxiv.org/abs/2511.20123), which achieve diffusion-transformer extrapolation for long video generation in a plug-and-play way.

<div align="center">
<img src='assets/riflex.png'></img>

<a href='https://arxiv.org/pdf/2502.15894'><img src='https://img.shields.io/badge/arXiv-2502.15894-b31b1b.svg'></a>
<a href='https://arxiv.org/pdf/2511.20123'><img src='https://img.shields.io/badge/arXiv-2511.20123-b31b1b.svg'></a> 
</div>

This repository hosts RIFLEx and UltraViCo on separate branches, and the code is fully open source.

- RIFLEx: 
    - [main](https://github.com/thu-ml/DiT-Extrapolation): HunyuanVideo-diffusers and CogVideoX-diffusers
    - [multi-gpu](https://github.com/thu-ml/DiT-Extrapolation/tree/multi-gpu): multi-GPU inference for HunyuanVideo

- UltraViCo:
    - [ultra-wan](https://github.com/thu-ml/DiT-Extrapolation/tree/ultra-wan): UltraViCo for **Wan2.1**
    - [ultra-hunyuan](https://github.com/thu-ml/DiT-Extrapolation/tree/ultra-hunyuan):UltraViCo for **HunyuanVideo**

---
<div align="center">

## UltraViCo: Breaking Extrapolation Limits in Video Diffusion Transformers
<a href='https://thu-ml.github.io/UltraViCo.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 
<a href='https://arxiv.org/abs/2511.20123'><img src='https://img.shields.io/badge/arXiv-22511.20123-b31b1b.svg'></a> <br>
<div class="is-size-6 publication-authors">
              <span class="author-block">
                <a href="https://gracezhao1997.github.io/">Min Zhao*</a>,
              </span>
              <span class="author-block">
                <a href="https://zhuhz22.github.io/">Hongzhou Zhu*</a>,
              </span>
              <span class="author-block">
                <a href="https://voyagerthu.github.io/minimal-light/">Yingze Wang</a>,
              </span>
              <span class="author-block">
                <a href="https://github.com/KasenYoung">Bokai Yan</a>,
              </span>
              <span class="author-block">
                <a href="https://gracezhao1997.github.io/">Jintao Zhang</a>,
              </span>
              <span class="author-block">
                <a href="https://guandehe.github.io/">Guande He</a>,
              </span>
              <span class="author-block">
                <a href="https://zhenxuan00.github.io/">Ling Yang</a>,
              </span>
              <span class="author-block">
                <a href="https://zhenxuan00.github.io/">Chongxuan Li</a>,
              </span>
              <span class="author-block">
                <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml">Jun Zhu</a>
              </span>
            </div>
</div>
<br>
This branch supports UltraViCo for HunyuanVideo. For Wan 2.1, please refer to the `ultra-wan` branch.

### Installation

```bash
conda create -n ultravico_hy python=3.11 -y
conda activate ultravico_hy
pip install -r requirements.txt
```

### Inference

```bash
export PYTHONPATH=$(pwd)/src

torchrun --nproc_per_node=8 --standalone -m parallel_examples.run_attention_patterns \
  --alpha 0.9 \
  --beta 0.6 \
  --extrapolation_ratio 3 \
  --height 544 \
  --width 960 \
  --num_inference_steps 50 \
  --prompt "Brown bear wading slowly through shallow river, splashes frozen mid-air, forest reflection steady on water surface."
```

- `extrapolation_ratio` $\in (1,4]$ : the generated video length as a multiple of the training length

- `alpha` < `beta` $\in (0,1)$: larger → stronger temporal consistency; smaller → better visual quality.

### Acknowledge

- We adopt [SageAttention](https://github.com/thu-ml/SageAttention) for our UltraViCo attention kernel.

- The code of the parallel diffusers-style HunyuanVideo is build upon [ParaAttention](https://github.com/chengzeyi/ParaAttention). Many tanks to its developers!

- Thank [Tecent HuyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) and [Wan2.1](https://github.com/Wan-Video/Wan2.1) for their great open-source models!

---

## References
If you find the code useful, please cite
```bibtex

@article{zhao2025ultravico,
  title={UltraViCo: Breaking Extrapolation Limits in Video Diffusion Transformers},
  author={Zhao, Min and Zhu, Hongzhou and Wang, Yingze and Yan, Bokai and Zhang, Jintao and He, Guande and Yang, Ling and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2511.20123},
  year={2025}
}

@article{zhao2025riflex,
  title={Riflex: A free lunch for length extrapolation in video diffusion transformers},
  author={Zhao, Min and He, Guande and Chen, Yixiao and Zhu, Hongzhou and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2502.15894},
  year={2025}
}
```
