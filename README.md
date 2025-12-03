## Diffusion-Transformer Extrapolation for Long Video Generation
This repository provides the official implementation of [RIFLEx](https://arxiv.org/abs/2502.15894) and [UltraViCo](https://arxiv.org/abs/2511.20123), which achieve diffusion-transformer extrapolation for long video generation in a plug-and-play way.

<div align="center">
<img src='assets/riflex.png'></img>

<a href='https://arxiv.org/pdf/2502.15894'><img src='https://img.shields.io/badge/arXiv-2502.15894-b31b1b.svg'></a>
<a href='https://arxiv.org/pdf/2511.20123'><img src='https://img.shields.io/badge/arXiv-2511.20123-b31b1b.svg'></a> 
</div>

This repository hosts RIFLEx and UltraViCo on separate branches, and the code is fully open source.

- RIFLEx: 
    - `main`: HunyuanVideo-diffusers and CogVideoX-diffusers
    - `multi-gpu`: multi-GPU inference for HunyuanVideo

- UltraViCo:
    - `ultra-wan`: UltraViCo for **Wan2.1**
    - `ultra-hunyuan`:UltraViCo for **HunyuanVideo**

---
<div align="center">

## RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers
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
