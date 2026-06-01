# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch.nn as nn
import math
import torch
import pywt
import torch.fft as fft
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

import torch
import torch.nn.functional as F

def make_rect_cosine_mask(H, W, inner_ratio=0.6, device="cpu"):
    """
    生成中心为1、边缘余弦衰减到0的矩形mask。
    用于outpainting区域平滑过渡。
    """
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    dist_x = torch.abs(xx)
    dist_y = torch.abs(yy)

    def cosine_weight(dist):
        inner = inner_ratio
        outer = 1.0
        w = torch.zeros_like(dist)
        inside = dist <= inner
        smooth = (dist > inner) & (dist < outer)
        w[inside] = 1.0
        w[smooth] = 0.5 * (1 + torch.cos(torch.pi * (dist[smooth] - inner) / (outer - inner)))
        return w

    weight_x = cosine_weight(dist_x)
    weight_y = cosine_weight(dist_y)
    mask = torch.minimum(weight_x, weight_y)
    return mask

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""

def split_frequency_components_dwt(x, wavelet='haar', level=1):
    device = x.device
    dtype = x.dtype
    x = x.type(torch.float32)
    x = x.cpu().numpy()

    B, C, H, W = x.shape
    low_freq_components = []

    # Using list comprehension to improve performance
    for b in range(B):
        for c in range(C):
            coeffs = pywt.wavedec2(x[b, c], wavelet=wavelet, level=level)
            low_freq, *high_freq = coeffs
            low_freq_components.append([low_freq] + [(np.zeros_like(detail[0]), np.zeros_like(detail[1]), np.zeros_like(detail[2])) for detail in high_freq])

    # Convert list of numpy arrays to a single numpy array for better performance
    x_low_freq = np.stack([pywt.waverec2(low_freq_components[i], wavelet=wavelet) for i in range(B * C)])

    # Convert the numpy array to a tensor
    x_low_freq = torch.from_numpy(x_low_freq).view(B, C, H, W).type(dtype).to(device)

    return x_low_freq

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    # to do: max_shift: float = 1.15,
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# class approx_midpoint:
#     def __init__(self):
#         self.


class FluxPipeline(DiffusionPipeline, FluxLoraLoaderMixin):
    r"""
    The Flux pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 64
        # to do: self.default_sample_size = 128

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    @staticmethod
    def _unpack_latents_hz(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = height // vae_scale_factor
        width = width // vae_scale_factor
        # TODO: hz modified here, currently hard-coded. 
        latents = latents.view(batch_size, 64, 64, channels // 4, 2, 2) # [1, xx, xx, 16,2,2]
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        # TODO: hz modified here, currently hard-coded. 
        latents = latents.reshape(batch_size, channels // (2 * 2), 64 * 2, 64 * 2)

        return latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        prompt_high: Union[str, List[str]] = None,
        prompt_high_2: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        timesteps: List[int] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        #########################
        num_inference_steps1: int = 30,
        num_inference_steps2: int = 20,
        guidance_scale1: float = 3.5,
        guidance_scale2: float = 7.0,
        ntk_factor: float = 10.0,
        time_shift_1: float = 3.0,
        time_shift_2: float = 6.0,
        proportional_attention = True,
        text_duplication = True,
        swin_pachify = True,
        dwt_level: int = 1,
        guidance_schedule = "cosine_shift",
        sdedit_threshold: int = 990,
        native_resolution = 1024,
        start_croords = [0.5,0.5], #左上角坐标, 0-1之间，放在图像的哪个位置
        hyper_params: Dict[str, Any] = {},
        method: Optional[Dict[str, Any]] = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        scale_factor = math.sqrt((height * width) / native_resolution ** 2)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale1
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        (
            prompt_embeds_highres,
            pooled_prompt_embeds_highres,
            text_ids_highres,
        ) = self.encode_prompt(
            prompt=prompt_high,
            prompt_2=prompt_high_2,
            prompt_embeds=None if 'outpaint' in guidance_schedule else prompt_embeds,
            pooled_prompt_embeds=None if 'outpaint' in guidance_schedule else pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            native_resolution, # to do: height
            native_resolution, # to do: width
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps1, num_inference_steps1)
        image_seq_len = latents.shape[1]
        self.scheduler.config.shift = time_shift_1
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps1 = retrieve_timesteps(
            self.scheduler,
            num_inference_steps1,
            device,
            timesteps, # to do: timesteps
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps1 * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # to do: self.scheduler.set_begin_index(0)
        # 6.1 Denoising loop (Native Resolution) ###########################################################################
        # note that generate naive resolution does not require 'method'. 
        with self.progress_bar(total=num_inference_steps1) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # handle guidance
                if self.transformer.config.guidance_embeds:
                    guidance = torch.tensor([guidance_scale1], device=device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None
                noise_pred = self.transformer(
                    hidden_states=latents,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    ntk_factor=1,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 6.3 Upsampling ###########################################################################
        if guidance_schedule == "super_imax":
            # print(f'latent.shape is {latents.shape}')
            # print(f'Begin: height is {height}, width is {width}')
            latents = self._unpack_latents_hz(latents, height // int(scale_factor + 0.5), width // int(scale_factor + 0.5), self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image_guidance = self.vae.tiled_decode(latents, return_dict=False)[0]
            image_guidance = torch.nn.functional.interpolate(image_guidance, size=(height, width), mode='bicubic')
            latents = self.vae.tiled_encode(image_guidance).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            latents_guidance = self._pack_latents(latents, batch_size, num_channels_latents, height // self.vae_scale_factor * 2, width // self.vae_scale_factor * 2)
        else:
            latents_guidance = latents
            shape = latents_guidance.shape
            noise = randn_tensor(shape, generator=generator, device=device)

        # 6.3 Re-setting ###########################################################################
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None, # to do: latents
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps2, num_inference_steps2)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        self.scheduler.config.shift = time_shift_2
        timesteps_extra, num_inference_steps2 = retrieve_timesteps(
            self.scheduler,
            num_inference_steps2,
            device,
            timesteps=None,
            sigmas=sigmas,
            mu=mu,
        )

        # self.scheduler.set_begin_index(0)
        # 6.4 Denoising loop (Extrapolated Resolution) ###########################################################################
        with self.progress_bar(total=num_inference_steps2) as progress_bar:
            for i, t in enumerate(timesteps_extra):
                if self.interrupt:
                    continue

                if swin_pachify and guidance_schedule == "super_imax":
                    if i % 2 == 0:
                        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                        latents = F.pad(latents, pad=(1, 1, 1, 1), mode='constant', value=0)
                        latents = self._pack_latents(latents, batch_size, num_channels_latents, height // self.vae_scale_factor * 2 + 2, width // self.vae_scale_factor * 2 + 2)
                        latents_guidance = self._unpack_latents(latents_guidance, height, width, self.vae_scale_factor)
                        latents_guidance = F.pad(latents_guidance, pad=(1, 1, 1, 1), mode='constant', value=0)
                        latents_guidance = self._pack_latents(latents_guidance, batch_size, num_channels_latents, height // self.vae_scale_factor * 2 + 2, width // self.vae_scale_factor * 2 + 2)
                        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // self.vae_scale_factor * 2 + 2, width // self.vae_scale_factor * 2 + 2, device, latents.dtype)
                        height = height + 1 * self.vae_scale_factor
                        width = width + 1 * self.vae_scale_factor
                    else:
                        height = height - 1 * self.vae_scale_factor
                        width = width - 1 * self.vae_scale_factor
                        latents = self._unpack_latents(latents, height + 1 * self.vae_scale_factor, width + 1 * self.vae_scale_factor, self.vae_scale_factor)
                        latents = latents[:, :, 1:-1, 1:-1]
                        latents = self._pack_latents(latents, batch_size, num_channels_latents, height // self.vae_scale_factor * 2, width // self.vae_scale_factor * 2)
                        latents_guidance = self._unpack_latents(latents_guidance, height + 1 * self.vae_scale_factor, width + 1 * self.vae_scale_factor, self.vae_scale_factor)
                        latents_guidance = latents_guidance[:, :, 1:-1, 1:-1]
                        latents_guidance = self._pack_latents(latents_guidance, batch_size, num_channels_latents, height // self.vae_scale_factor * 2, width // self.vae_scale_factor * 2)
                        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // self.vae_scale_factor * 2, width // self.vae_scale_factor * 2, device, latents.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # handle guidance
                if self.transformer.config.guidance_embeds:
                    guidance = torch.tensor([guidance_scale2], device=device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None
                
                if i < sdedit_threshold and guidance_schedule == "outpaint_sdedit":
                    t_step = t.expand(latents.shape[0])
                    gt_t = self.scheduler.scale_noise(latents_guidance, t_step, noise)
                    gt_t = self._unpack_latents(gt_t, native_resolution, native_resolution, self.vae_scale_factor)
                    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                    gt_height, gt_width = gt_t.shape[2], gt_t.shape[3]
                    latent_height,latent_width = latents.shape[2],latents.shape[3]
                    start_x = min(max(int(start_croords[0] * latent_height) - gt_height//2, 0), latent_height - gt_height)
                    start_y = min(max(int(start_croords[1] * latent_width) - gt_width//2, 0), latent_width - gt_width)

                    latents[:,:, start_y:start_y+gt_height, start_x:start_x + gt_width] = gt_t
                    latents = self._pack_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)

                if i < sdedit_threshold and guidance_schedule == "outpaint_sdedit_weight":
                    inner_ratio = hyper_params.get("inner_ratio", 0.9) 
                    t_step = t.expand(latents.shape[0])
                    gt_t = self.scheduler.scale_noise(latents_guidance, t_step, noise)
                    gt_t = self._unpack_latents(gt_t, native_resolution, native_resolution, self.vae_scale_factor)
                    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)       
                    gt_height, gt_width = gt_t.shape[2], gt_t.shape[3]
                    latent_height, latent_width = latents.shape[2], latents.shape[3]
                    start_x = min(max(int(start_croords[0] * latent_height) - gt_height // 2, 0), latent_height - gt_height)
                    start_y = min(max(int(start_croords[1] * latent_width) - gt_width // 2, 0), latent_width - gt_width)
                    blend_mask = make_rect_cosine_mask(gt_height, gt_width, inner_ratio=inner_ratio, device=latents.device)
                    blend_mask = blend_mask[None, None, :, :]  # shape: [1,1,H,W]
                    region = latents[:, :, start_y:start_y + gt_height, start_x:start_x + gt_width]
                    latents[:, :, start_y:start_y + gt_height, start_x:start_x + gt_width] = \
                        (1 - blend_mask) * region + blend_mask * gt_t
                    latents = self._pack_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds_highres,
                    encoder_hidden_states=prompt_embeds_highres,
                    txt_ids=text_ids_highres,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    ntk_factor=ntk_factor,
                    proportional_attention=proportional_attention,
                    text_duplication=text_duplication,
                    method=method,
                )[0]
                
                if guidance_schedule == "outpaint_imax":
                    latents_unpack = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                    latents_guidance_unpack = self._unpack_latents(latents_guidance, native_resolution, native_resolution, self.vae_scale_factor)
                    noise_pred_ = self._unpack_latents(noise_pred, height, width, self.vae_scale_factor)
                    latent_height, latent_width = noise_pred_.shape[2],noise_pred_.shape[3]
                    guidance_height, guidance_width = int(latent_height//scale_factor), int(latent_width//scale_factor)
                    start_x = min(max(int(start_croords[0] * latent_height) - guidance_height//2, 0), latent_height - guidance_height)
                    start_y = min(max(int(start_croords[1] * latent_width) - guidance_width//2, 0), latent_width - guidance_width)
                    latents_outpaint = latents_unpack[:,:, start_y:start_y+guidance_height, start_x:start_x + guidance_width]
                    noise_pred_outpaint = noise_pred_[:,:, start_y:start_y+guidance_height, start_x:start_x + guidance_width]
                    fp_v_ = -(latents_guidance_unpack - latents_outpaint) / (t / 1000 + 1e-6)
                    weight_factor = 1.0
                    step_i = torch.tensor(i / num_inference_steps2, device=latents.device, dtype=latents.dtype)
                    weight_schedule = hyper_params.get("weight_schedule", "cosine")
                    if weight_schedule == "cosine":
                        weight_factor = 0.5 * (1 + torch.cos(step_i))
                    elif weight_schedule == "linear":
                        weight_factor = 1 - step_i
                    elif weight_schedule == "exp":
                        weight_factor = torch.exp(-5 * step_i)
                    elif weight_schedule == "power":
                        weight_factor = (1 - step_i)**2
                    elif weight_schedule == "logistic":
                        weight_factor = 1 / (1 + torch.exp(10*(step_i - 0.5)))

                    fp_v_low = split_frequency_components_dwt(fp_v_, level=dwt_level)
                    noise_pred_outpaint_low = split_frequency_components_dwt(noise_pred_outpaint, level=dwt_level)

                    noise_pred_outpaint_guidance = noise_pred_outpaint + weight_factor * (fp_v_low - noise_pred_outpaint_low)
                    noise_pred_[:,:, start_y:start_y+guidance_height, start_x:start_x + guidance_width] = noise_pred_outpaint_guidance
                    noise_pred = self._pack_latents(noise_pred_, batch_size, num_channels_latents, latent_height, latent_width)

                if guidance_schedule == "super_imax":
                    fp_v = -(latents_guidance - latents) / (t / 1000 + 1e-6)
                    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (i / torch.Tensor([num_inference_steps2])))).to(latents.device, latents.dtype)
                    fp_v_ = self._unpack_latents(fp_v, height, width, self.vae_scale_factor)
                    noise_pred_ = self._unpack_latents(noise_pred, height, width, self.vae_scale_factor)
                    fp_v_low = split_frequency_components_dwt(fp_v_, level=dwt_level)
                    noise_pred_low = split_frequency_components_dwt(noise_pred_, level=dwt_level)
                    fp_v_low = self._pack_latents(fp_v_low, batch_size, num_channels_latents, height // self.vae_scale_factor * 2, width // self.vae_scale_factor * 2)
                    noise_pred_low = self._pack_latents(noise_pred_low, batch_size, num_channels_latents, height // self.vae_scale_factor * 2, width // self.vae_scale_factor * 2)
                    noise_pred = noise_pred + cosine_factor * (fp_v_low - noise_pred_low)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # 

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds_highres = callback_outputs.pop("prompt_embeds", prompt_embeds_highres)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            # image = self.vae.decode(latents, return_dict=False)[0]
            image = self.vae.tiled_decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

            if guidance_schedule != "super_imax":
                latents_guidance = self._unpack_latents(latents_guidance, native_resolution, native_resolution, self.vae_scale_factor)
                latents_guidance = (latents_guidance / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image_guidance = self.vae.tiled_decode(latents_guidance, return_dict=False)[0]
                image_guidance = self.image_processor.postprocess(image_guidance, output_type=output_type)
            else:
                # For super_imax, image_guidance was already decoded at line 766-767 but needs postprocessing
                image_guidance = self.image_processor.postprocess(image_guidance, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image + image_guidance

        return FluxPipelineOutput(images=image)
