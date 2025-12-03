import importlib

from diffusers import DiffusionPipeline


def parallelize_transformer(transformer, *args, **kwargs):
    transformer_cls_name = transformer.__class__.__name__
    if False:
        pass
    elif transformer_cls_name.startswith("Flux"):
        adapter_name = "flux"
    elif transformer_cls_name.startswith("Mochi"):
        adapter_name = "mochi"
    elif transformer_cls_name.startswith("CogVideoX"):
        adapter_name = "cogvideox"
    elif transformer_cls_name.startswith("HunyuanVideo"):
        adapter_name = "hunyuan_video"
    elif transformer_cls_name.startswith("Wan"):
        adapter_name = "wan"
    else:
        raise ValueError(f"Unknown transformer class name: {transformer_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    parallelize_transformer_fn = getattr(adapter_module, "parallelize_transformer")
    return parallelize_transformer_fn(transformer, *args, **kwargs)


def parallelize_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    assert isinstance(pipe, DiffusionPipeline)

    pipe_cls_name = pipe.__class__.__name__
    if False:
        pass
    elif pipe_cls_name.startswith("Flux"):
        adapter_name = "flux"
    elif pipe_cls_name.startswith("Mochi"):
        adapter_name = "mochi"
    elif pipe_cls_name.startswith("CogVideoX"):
        adapter_name = "cogvideox"
    elif pipe_cls_name.startswith("HunyuanVideo"):
        adapter_name = "hunyuan_video"
    elif pipe_cls_name.startswith("Wan"):
        adapter_name = "wan"
    else:
        raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    parallelize_pipe_fn = getattr(adapter_module, "parallelize_pipe")
    return parallelize_pipe_fn(pipe, *args, **kwargs)
