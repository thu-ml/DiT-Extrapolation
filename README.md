<div align="center">
<img src='assets/ultraimage.png'></img>
</div>

## UltraImage: Rethinking Resolution Extrapolation in Image Diffusion Transformers
<a href='https://thu-ml.github.io/ultraimage.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 
<a href='https://arxiv.org/pdf/2512.04504'><img src='https://img.shields.io/badge/arXiv-2512.04504-b31b1b.svg'></a> <br>
<div class="is-size-6 publication-authors">
              <span class="author-block">
                <a href="https://gracezhao1997.github.io/">Min Zhao</a>,
              </span>
              <span class="author-block">
                <a href="https://github.com/KasenYoung">Bokai Yan</a>,
              </span>
              <span class="author-block">
                <a href="https://github.com/Sakura4111">Xue Yang</a>,
              </span>
              <span class="author-block">
                <a href="https://zhuhz22.github.io/">Hongzhou Zhu</a>,
              </span>
              <span class="author-block">
                <a href="https://scholar.google.com/citations?user=J3NTnmEAAAAJ&hl=zh-CN&oi=sra">Jintao Zhang</a>,
              </span>
              <span class="author-block">
                <a >Shilong Liu</a>,
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
This branch supports UltraImage for Qwen-Image. For Flux, please refer to the `ultra-flux` branch.

### Installation

```bash
cd qwen
conda create -n QwenImage python=3.10 -y
conda activate QwenImage
pip install -r requirements_qwen.txt
```

### Inference

```bash
CUDA_VISIBLE_DEVICES={gpu id} python inference_rope.py --config {your config path}
```
You can download the model [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) and place it in your directory.

We provide demo config in base.yaml in the directory. Please modify your model_path, prompt_path, output_dir and save_dir for block average corresponding to your device.

### Acknowledge

- We adopt [SageAttention](https://github.com/thu-ml/SageAttention) for our UltraImage attention kernel.

- We acknowledge the contribution of [FLUX.1](https://github.com/black-forest-labs/flux) and [Qwen-Image](https://github.com/QwenLM/Qwen-Image) for their great open-source model, and [Diffusers](https://github.com/huggingface/diffusers) for their great open-source framework.


## References
If you find the code useful, please cite
```bibtex

@article{zhao2025ultraimage,
  title={UltraImage: Rethinking Resolution Extrapolation in Image Diffusion Transformers},
  author={Zhao, Min and Yan, Bokai and Yang, Xue and Zhu, Hongzhou and Zhang, Jintao and Liu, Shilong and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2512.04504},
  year={2025}
}
```
