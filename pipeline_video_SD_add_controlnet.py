# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import datetime
import numpy as np
import PIL.Image
import torch
from jupyterlab.handlers.announcements import news_handler_path
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AsymmetricAutoencoderKL, AutoencoderKL, UNet2DConditionModel,ControlNetModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers,PIL_INTERPOLATION
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import gc

# *************************************************************

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pylab
from PIL import Image
import math
from einops import rearrange, repeat
import os
import cv2

import shutil

import torch.nn.functional as F


tensor = transforms.ToTensor()


def read_jpg_images_path(folder_path,img_type):
    image_paths = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(img_type):
            image_path = os.path.join(folder_path, file_name)
            image_paths.append(image_path)
    return image_paths



def tensor_roll(x, offset_h=None, offset_w=None, padding=0):
    s = torch.clone(x)
    if offset_h is not None:
        if offset_h > 0:
            s[:, :, :offset_h, :] = padding
        elif offset_h < 0:
            s[:, :, offset_h:, :] = padding
    if offset_w is not None:
        if offset_w > 0:
            s[:, :, :, :offset_w] = padding
        elif offset_w < 0:
            s[:, :, :, offset_w:] = padding
    return torch.roll(s,(0, 0, -offset_h, -offset_w), (0,1,2,3))


def motion_fun(image,x_scale, y_scale, height=512, width=512):
    dx = x_scale*width  # 水平平移量
    dy = y_scale*height  # 垂直平移量
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    translated_image = image.transform(image.size, Image.AFFINE, M.flatten()[:6], resample=Image.BICUBIC)
    return translated_image


def set_background_mask(direction, x_scale, y_scale, height=512, width=512):

    x = x_scale

    y = y_scale
    arr = np.zeros((width, height))

    if direction == "left up":
        arr[:x, :] = 1
        arr[:, :y] = 1

    if direction == "right up":
        arr[:x, :] = 1
        arr[:, width - y:] = 1

    if direction == "left down":
        arr[height - x:, :] = 1
        arr[:, :y] = 1

    if direction == "right down":
        arr[height - x:, :] = 1
        arr[:, width - y:] = 1

    return tensor(arr)


def set_object_mask(mask_h,mask_w,height=512, width=512):
    arr = np.zeros((width, height))
    a = math.floor((height-mask_h)/2)
    b = math.floor((height+mask_h)/2)
    c = math.floor((width-mask_w)/2)
    d = math.floor((width+mask_w)/2)
#     arr[(height-mask_h)/2:(height+mask_h)/2,(width-mask_w)/2:(width+mask_w)/2] = 1
    arr[a:b,c:d] = 1
    return tensor(arr)






















logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    deprecation_message = "The prepare_mask_and_masked_image method is deprecated and will be removed in a future version. Please use VaeImageProcessor.preprocess instead"
    deprecate(
        "prepare_mask_and_masked_image",
        "0.30.0",
        deprecation_message,
    )
    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
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
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class controlnet_inpaint_video_pipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, IPAdapterMixin, LoraLoaderMixin, FromSingleFileMixin
):
    r"""
    Pipeline for text-guided image inpainting using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`, `AsymmetricAutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "mask", "masked_image_latents"]

    def __init__(
        self,
        vae: Union[AutoencoderKL, AsymmetricAutoencoderKL],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "skip_prk_steps") and scheduler.config.skip_prk_steps is False:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration"
                " `skip_prk_steps`. `skip_prk_steps` should be set to True in the configuration file. Please make"
                " sure to update the config accordingly as not setting `skip_prk_steps` in the config might lead to"
                " incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face"
                " Hub, it would be very nice if you could open a Pull request for the"
                " `scheduler/scheduler_config.json` file"
            )
            deprecate("skip_prk_steps not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["skip_prk_steps"] = True
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        # Check shapes, assume num_channels_latents == 4, num_channels_mask == 1, num_channels_masked == 4
        if unet.config.in_channels != 9:
            logger.info(f"You have loaded a UNet with {unet.config.in_channels} input channels which.")

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        strength,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

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
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )



    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb



    # copy from pipeline_stable_diffusion_controlnet
    # prepare  pose image

    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                image = [
                    np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

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
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs


    def prepare_latents_fb(self, batch_size, num_channels_latents, height, width, dtype, device, generator,
                        latents=None, frame_same_noise=False):

        shape = (
            batch_size, num_channels_latents, height // self.vae_scale_factor,
            width // self.vae_scale_factor)
        if frame_same_noise:
            shape = (1, num_channels_latents, height // self.vae_scale_factor,
                     width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
            if frame_same_noise:
                latents = repeat(latents, '1 c h w -> b c h w', b=batch_size)

        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    @torch.no_grad()
    def inference(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        image_pose: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        controlnet_conditioning_scale: float = 1.0,
        bg_theta = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be inpainted (which parts of the image to
                be masked out with `mask_image` and repainted according to `prompt`). For both numpy array and pytorch
                tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the
                expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the
                expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image latents as `image`, but
                if passing latents directly it is not encoded again.
            mask_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
                are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
                color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
                H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
                1)`, or `(H, W)`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
        Examples:

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionInpaintPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

        >>> init_image = download_image(img_url).resize((512, 512))
        >>> mask_image = download_image(mask_url).resize((512, 512))

        >>> pipe = StableDiffusionInpaintPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        >>> image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        if image_pose is not None:
            # 4. Prepare pose image
            image_pose = self.prepare_image(
                image_pose,
                width,
                height,
                batch_size * num_images_per_prompt,
                num_images_per_prompt,
                device,
                prompt_embeds.dtype,
            )
            if self.do_classifier_free_guidance:
                if image_pose.ndim == 4:
                    image_pose = torch.cat([image_pose]*2)
                else:
                    image_pose = torch.unsqueeze(image_pose, dim=0)
                    image_pose = torch.cat([image_pose] * 2)



        # 5. Preprocess mask and image

        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4


        x_base = self.prepare_latents_fb(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
            frame_same_noise=True)

        x_res = self.prepare_latents_fb(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
            frame_same_noise=False)





        if bg_theta is not None:
            for i, value in enumerate(bg_theta):

                # 使用最大池化操作，每个 8x8 的区域取最大值
                bbbb = F.max_pool2d(mask_image[i, :, :].unsqueeze(0), kernel_size=8, stride=8)
                # bbbb = bbbb.squeeze(0).numpy()
                # 使用 Matplotlib 显示
                # plt.imshow(bbbb, cmap='gray')  # 灰度图像
                # plt.axis('off')  # 隐藏坐标轴
                # plt.show()
                mask_image_temp = F.max_pool2d(mask_image.sum(dim=0).unsqueeze(0), kernel_size=8, stride=8)




                if latents is None:
                    latents = x_base * (1-mask_image_temp).to(device=self._execution_device,dtype=x_base.dtype)

                    a = latents[0, 0, :, :].cpu().numpy()
                    plt.imshow(a, cmap='gray')  # 灰度图像
                    plt.axis('off')  # 隐藏坐标轴
                    plt.show()
                    a = 1


                    latents += (np.cos(value * np.pi / 2) * x_base + np.sin(value * np.pi / 2) * x_res) * bbbb.to(device=self._execution_device,dtype=x_base.dtype)
                else:
                    latents += (np.cos(value * np.pi / 2) * x_base + np.sin(value * np.pi / 2) * x_res) * bbbb.to(device=self._execution_device,dtype=x_base.dtype)

                a = latents[0,0,:,:].cpu().numpy()
                plt.imshow(a, cmap='gray')  # 灰度图像
                plt.axis('off')  # 隐藏坐标轴
                plt.show()
                a= 1
        else:
            latents = x_base

        # 7. Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(mask_image.sum(dim=0), height=height, width=width)









        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 9.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if image_pose is not None:
                    control_model_input = latent_model_input

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)




                if image_pose is not None:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=image_pose,
                        return_dict=False,
                    )

                    down_block_res_samples = [
                        down_block_res_sample * controlnet_conditioning_scale
                        for down_block_res_sample in down_block_res_samples
                    ]
                    mid_block_res_sample *= controlnet_conditioning_scale
                else:
                    down_block_res_samples = None
                    mid_block_res_sample = None



                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                # if num_channels_unet == 4:
                #     init_latents_proper = image_latents
                #     if self.do_classifier_free_guidance:
                #         init_mask, _ = mask.chunk(2)
                #     else:
                #         init_mask = mask
                #
                #     if i < len(timesteps) - 1:
                #         noise_timestep = timesteps[i + 1]
                #         init_latents_proper = self.scheduler.add_noise(
                #             init_latents_proper, noise, torch.tensor([noise_timestep])
                #         )
                #
                #     latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    mask = callback_outputs.pop("mask", mask)
                    masked_image_latents = callback_outputs.pop("masked_image_latents", masked_image_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            condition_kwargs = {}
            if isinstance(self.vae, AsymmetricAutoencoderKL):
                init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
                init_image_condition = init_image.clone()
                init_image = self._encode_vae_image(init_image, generator=generator)
                mask_condition = mask_condition.to(device=device, dtype=masked_image_latents.dtype)
                condition_kwargs = {"image": init_image_condition, "mask": mask_condition}
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False, generator=generator, **condition_kwargs
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            if output_type == "tensor":
                return image
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)











    @torch.no_grad()
    def __call__(
        self,
        prompt_bg,
        prompt_obj,
        frame_num,
        save_path,
        image,
        image_pose_paths,
        seed_bg,
        seed_obj,
        Maskingapp,
        CurveDrawingApp,
        RectangleDrawerApp,
        crop_size,
        negative_prompt_bg,
        negative_prompt_obj,
        config_file,
        device="cuda"
    ):

        generator = torch.Generator(device=device)
        generator.manual_seed(seed_bg)
        import utils
        self.unet.set_attn_processor(
            processor=utils.CrossFrameAttnProcessor(unet_chunk_size=2, alpha=None))

        import re
        top_folder_name = prompt_obj[0][:20] + '___' + prompt_bg[0][:20]
        top_folder_name = re.sub(r'[^\w\-]', '_', top_folder_name)  # 非字母数字或下划线/横杠替换成 _

        exp_path = os.path.join(save_path, top_folder_name)
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        full_path = os.path.join(exp_path,now)
        source_path = os.path.join(exp_path,now,'source')
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            os.makedirs(source_path)
            print(f"文件夹 '{full_path}' 创建成功！")
            print(f"文件夹 '{source_path}' 创建成功")

        log_file = f"{full_path}/log.txt"

        # 复制文件到目标文件夹
        # 这里使用 os.path.basename 获取源文件的文件名
        shutil.copy(config_file, os.path.join(source_path, os.path.basename(config_file)))



        image_BG_temp = None
        # ****************************************** 原始背景 *****************************************
        # ********** 提供全部背景帧 ************
        # -- 当提供所有背景帧时，直接进入前景对象生成流程 -- #
        if isinstance(image,list):
            mode = 'list'
            # for i in range(frame_num):
            #     image_temp = torch.unsqueeze(tensor(image[i]), dim=0)
            #     if image_BG_temp==None:
            #         image_BG_temp = image_temp
            #     else:
            #         image_BG_temp = torch.cat([image_BG_temp, image_temp])
            # image_BG_temp = image_BG_temp.to(device=self._execution_device)
            image_BG_temp = image
            BG_image_1 = image[0]

        # ********** 不提供背景帧 ************
        elif image is None:
            mode = 'no'
            mask_BG_full = set_background_mask(direction="right up", x_scale=512, y_scale=512, height=512,
                                          width=512)
            mask_BG_full = mask_BG_full.to(device=self._execution_device)

            numpy_data = np.random.rand(3, 512, 512)
            noise_image = torch.tensor(numpy_data)
            noise_image = torch.unsqueeze(noise_image, dim=0)
            noise_image = self.image_processor.preprocess(noise_image, height=512, width=512)
            noise_image = noise_image.to(device=self._execution_device)

            BG_1 = self.inference(
                prompt=prompt_bg,
                image=noise_image,
                mask_image=mask_BG_full,
                generator=generator,
                negative_prompt=negative_prompt_bg,
                output_type="tensor"
            )
            BG_image_1 = self.image_processor.postprocess(BG_1, output_type="pil", do_denormalize=[True])[0]
            plt.imshow(BG_image_1)
            pylab.show()
        # ------------ 提供第一张原始背景 ---------#
        elif isinstance(image,PIL.Image.Image):
            BG_image_1 = image
            plt.imshow(BG_image_1)
            pylab.show()

        # ---------------------------------------------- 保存第一张原始背景 ------------------------ #
        step = 0
        group = 0
        BG_image_1.save(f'{full_path}/{step}_{group}_org_bg.png', format='PNG')

        gc.collect()
        torch.cuda.empty_cache()

        if isinstance(image,list):
            print('多背景模式，无须背景编辑')
        else:
            # ****************************************** 背景动态编辑 ***************************************** #
            # ****************************************电影编辑功能不需要此步骤
            maskingapp = Maskingapp(background_image_path=None,image=BG_image_1)
            weight_tensor,mask_data, masked_image, bg_masks, Noise_weight ,colors= maskingapp.run()

            log_Noise_weight = 'Noise_weight: ' + ', '.join(map(str, Noise_weight)) + '\n\n'
            log_colors = 'colors: ' + colors + '\n\n'
            with open(log_file, 'a') as file:
                file.write(log_Noise_weight)
                file.write(log_colors)



            plt.imshow(mask_data)
            pylab.show()
            plt.imshow(masked_image)
            pylab.show()

            # ------------------------------ 保存文件 ------------------- #
            step = step + 1
            group = 0
            Image.fromarray(mask_data).save(f'{full_path}/{step}_{group}_masked_data.png')
            masked_image.save(f'{full_path}/{step}_{group}_masked_org_bg.png', format='PNG')



            for i, a_lambda in enumerate(Noise_weight):
                mask_BG = torch.cat([torch.unsqueeze(bg_masks[i], dim=0)]*frame_num,dim=0)
                mask_BG = mask_BG.to(device=self._execution_device)

                plt.imshow(bg_masks[i], cmap='gray')  # 使用灰度色图显示
                plt.show()

            bg_masks = bg_masks.to(device=self._execution_device)
            image = torch.cat([torch.unsqueeze(tensor(BG_image_1), dim=0)]*frame_num)
            image = self.image_processor.preprocess(image, height=512, width=512)
            image = image.to(device=self._execution_device)
            prompt_bg = prompt_bg * frame_num

            image_BG_temp = self.inference(
                prompt=prompt_bg,
                image=image,
                mask_image=bg_masks,
                generator=generator,
                output_type="tensor",
                negative_prompt=negative_prompt_bg * frame_num,
                bg_theta=Noise_weight
            )
            image_BG_temp = self.image_processor.postprocess(image_BG_temp, output_type="pil", do_denormalize=[True]*frame_num)

            # ------------------------------ 保存文件 ------------------- #
            group = group + 1
            for k in range(frame_num):
                image_BG_temp[k].save(f'{full_path}/{step}_{group}_BGtemp_{k}.png', format='PNG')
                plt.imshow(image_BG_temp[k])
                pylab.show()

            group = group + 1
            # ----------------------------- 按区域保存 ----------------------#
            for i, bg in enumerate(image_BG_temp):
                plt.imshow(bg)
                pylab.show()
                bg_temp = bg.convert("RGBA")
                mask = mask_BG[i, :, :]

                bg_temp = tensor(bg_temp)
                # 应用掩膜
                bg_temp[3, :, :] = mask
                bg_temp = bg_temp * 255
                bg_temp_uint8 = bg_temp.byte()  # 转换为 uint8 类型

                # 转换为 numpy 数组并调整维度
                # 需要将通道数放在最后一维
                bg_image = bg_temp_uint8.permute(1, 2, 0).numpy()  # 变为 (512, 512, 4)
                # 创建 RGBA 图片
                bg_rgba = Image.fromarray(bg_image, mode='RGBA')

                # 创建一个新的 RGB 图像，背景为白色
                rgb_image = Image.new("RGB", bg_rgba.size, (255, 255, 255))

                # 将 ARGB 图像粘贴到新的 RGB 图像上，透明区域将显示为白色
                rgb_image.paste(bg_rgba, (0, 0), bg_rgba)

                plt.imshow(bg_rgba)
                pylab.show()

                rgb_image.save(f'{full_path}/{step}_{group}_BGregion_{i}.png', format='PNG')











        # ****************************************** 前景对象 batch_size = f *****************************************
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_obj)

        from utlis.orf_find import compute_optimal_reference_frame, load_pose_sequence_from_folder
        pose_sequence = load_pose_sequence_from_folder(image_pose_paths)
        alpha = compute_optimal_reference_frame(pose_sequence)
        import utils
        self.unet.set_attn_processor(
            processor=utils.CrossFrameAttnProcessor(unet_chunk_size=2, alpha=alpha))


        curvedrawingapp = CurveDrawingApp(repoint_num=frame_num,image_path=None,image=image_BG_temp[0])
        curve_point = curvedrawingapp.run()
        log_curve_point = 'log_curve_point: ' + ', '.join(map(str, curve_point)) + '\n\n'
        rectangledrawerapp = RectangleDrawerApp(image=image_BG_temp[0], initial_coords=curve_point)

        final_sizes = rectangledrawerapp.run()
        log_final_sizes = 'final_sizes: ' + ', '.join(map(str, final_sizes)) + '\n\n'


        with open(log_file, 'a') as file:
            file.write(log_curve_point)
            file.write(log_final_sizes)



        cropped_bg = []
        width, height = (crop_size, crop_size)
        for img, (x, y) ,width in zip(image_BG_temp, curve_point,final_sizes):
            # 计算截取区域
            food_offset = int(60*(width/512))
            box = (x-width//2, y-width+food_offset, x + width//2, y+food_offset)
            # 截取图片
            cropped_img = img.crop(box)
            cropped_bg.append(cropped_img.resize((512,512)))

        # ------------------------------ 保存文件 ------------------- #
        step = step + 1
        group = 0
        for k in range(frame_num):
            cropped_bg[k].save(f'{full_path}/{step}_{group}_cropped_BG_{k}.png', format='PNG')
            plt.imshow(image_BG_temp[k])
            pylab.show()

        BG_saved = self.image_processor.preprocess(cropped_bg, height=512, width=512)
        BG_saved = BG_saved.to(device=self._execution_device)

        for k in range(frame_num):
            plt.imshow(cropped_bg[k])
            pylab.show()

        if image_pose_paths is not None:
            pose_image_paths = read_jpg_images_path(folder_path=image_pose_paths, img_type='.png')


            # 膨胀卷积核
            kernel = np.ones((5, 5), np.uint8)
            # 读取每张pose图片并膨胀获得mask
            pose_masks = None
            pose_masks_mini = None
            pose_images = None
            for pose_image_path in pose_image_paths:
                pose_image = Image.open(pose_image_path)
                pose_image_tensor = torch.unsqueeze(tensor(pose_image), dim=0)

                # 膨胀处理，获得mask
                dilated_image = cv2.dilate(np.array(pose_image), kernel, iterations=10)
                pose_mask = dilated_image[:, :, 0] + dilated_image[:, :, 1] + dilated_image[:, :, 2]
                pose_mask[pose_mask > 0] = 255
                pose_mask_tensor = torch.unsqueeze(tensor(pose_mask), dim=0)

                # 膨胀处理，获得mask_mini
                dilated_image_mini = cv2.dilate(np.array(pose_image), kernel, iterations=7)
                pose_mask_mini = dilated_image_mini[:, :, 0] + dilated_image_mini[:, :, 1] + dilated_image_mini[:, :, 2]
                pose_mask_mini[pose_mask_mini > 0] = 255
                pose_mask_mini_tensor = torch.unsqueeze(tensor(pose_mask_mini), dim=0)


                if pose_images is None:
                    pose_images = pose_image_tensor
                    pose_masks = pose_mask_tensor
                    pose_masks_mini = pose_mask_mini_tensor
                else:
                    pose_images = torch.cat([pose_images, pose_image_tensor])
                    pose_masks = torch.cat([pose_masks,pose_mask_tensor])
                    pose_masks_mini = torch.cat([pose_masks_mini,pose_mask_mini_tensor])

            pose_images = pose_images.to(device=self._execution_device)
            pose_masks = pose_masks.to(device=self._execution_device)

        else:
            pose_masks = set_object_mask(mask_h=400, mask_w=400)
            pose_masks = torch.cat([torch.unsqueeze(pose_masks, dim=0)] * BG_saved.shape[0])
            pose_masks = pose_masks.to(device=self._execution_device)
            pose_images = None






        numpy_data = np.random.rand(BG_saved.shape[0],3, 512, 512)
        noise_image = torch.tensor(numpy_data)
        noise_image = self.image_processor.preprocess(noise_image, height=512, width=512)
        noise_image = noise_image.to(device=self._execution_device)






        prompt_obj = prompt_obj * BG_saved.shape[0]

        image_OUT = self.inference(
            prompt = prompt_obj,
            image = BG_saved,
            mask_image = pose_masks,
            image_pose=pose_images,
            generator = generator,
            negative_prompt=negative_prompt_obj * frame_num,
            output_type = "tensor"
        )


        org_img_obj = self.image_processor.postprocess(image_OUT, output_type="pil", do_denormalize=[True]*BG_saved.shape[0])

        # for i, (obj, (x, y),width) in enumerate(zip(org_img_obj, curve_point,final_sizes)):
        #     plt.imshow(obj)
        #     pylab.show()
        #     obj_temp = obj.convert("RGBA")
        #     mask = pose_masks_mini[i,0,:,:]
        #
        #     obj_temp = tensor(obj_temp)
        #     # 应用掩膜
        #     obj_temp[3, :, :] = mask
        #     obj_temp = obj_temp * 255
        #     obj_temp_uint8 = obj_temp.byte()  # 转换为 uint8 类型
        #
        #     # 转换为 numpy 数组并调整维度
        #     # 需要将通道数放在最后一维
        #     obj_image = obj_temp_uint8.permute(1, 2, 0).numpy()  # 变为 (512, 512, 4)
        #     # 创建 RGBA 图片
        #     obj_rgba = Image.fromarray(obj_image, mode='RGBA')
        #
        #     obj_rgba = obj_rgba.resize((width,width))
        #     plt.imshow(obj_rgba)
        #     pylab.show()
        #     food_offset = int(60 * (width / 512))
        #     # 将掩膜结果粘贴到目标图片上
        #     image_BG_temp[i].paste(obj_rgba, (x-width//2, y-width+food_offset), obj_rgba)  # 使用掩膜进行粘贴
        #     # image_BG_temp[i].paste(obj_temp, (x-width//2, y-width), obj_temp)  # 使用掩膜进行粘贴
        #     plt.imshow(image_BG_temp[i])
        #     pylab.show()




        for i, (obj, (x, y),width) in enumerate(zip(org_img_obj, curve_point,final_sizes)):
            plt.imshow(obj)
            pylab.show()
            obj_temp = obj.convert("RGBA")
            image_BG_temp[i] = image_BG_temp[i].convert("RGBA")

            # 指定缩放后的大小
            new_size = (width, width)  # 替换为你想要的大小

            # 缩放图像
            resized_image = obj_temp.resize(new_size, Image.ANTIALIAS)

            # 计算粘贴位置
            background_width, background_height = image_BG_temp[i].size
            food_offset = int(60 * (width / 512))
            paste_position = (x-width//2, y-width+food_offset)

            # 创建一个渐变蒙版
            mask_paste = Image.new("L", (new_size[0], new_size[1]), 0)  # 创建一个黑色蒙版
            for k in range(new_size[0]):
                for j in range(new_size[1]):
                    # 计算距离中心的距离
                    distance = min(k, j, new_size[0] - k, new_size[1] - j)
                    # 根据距离设置透明度，距离越近，透明度越低（越透明）
                    mask_paste.putpixel((k, j), int(255 * (distance / 20)))  # 调整20以控制渐变范围

            # 将缩放后的图像粘贴到背景图像上
            image_BG_temp[i].paste(resized_image, paste_position, mask_paste)

            plt.imshow(image_BG_temp[i])
            pylab.show()


        # -------------------- 保存文件 -------------------#
        group = group + 1
        for i in range(frame_num):
            org_img_obj[i].save(f'{full_path}/{step}_{group}_cropped_BG_obj_{i}.png', format='PNG')
            plt.imshow(org_img_obj[i])
            pylab.show()

        group = group + 1
        for i in range(frame_num):
            image_BG_temp[i].save(f'{full_path}/{step}_{group}_Result_{i}.png', format='PNG')
            plt.imshow(image_BG_temp[i])
            pylab.show()

        image_BG_temp[0].save(f'{full_path}/output.gif', format='GIF', append_images=image_BG_temp[1:], save_all=True, duration=500, loop=0)

        plt.close("all")
        torch.cuda.empty_cache()


        # # ****************************************** 前景对象 batch_size = 1 *****************************************
        # mask_OBJ = set_object_mask(mask_h=400, mask_w=400)
        # mask_OBJ = mask_OBJ.to(device=self._execution_device)
        #
        # result_obj_NoCrossAtte = None
        # for i in range(BG_saved.shape[0]):
        #     image_BG = self.inference(
        #         prompt=prompt_obj[0],
        #         image=BG_saved[i],
        #         mask_image=mask_OBJ,
        #         generator=generator,
        #         output_type="tensor"
        #     )
        #     if result_obj_NoCrossAtte is None:
        #         result_obj_NoCrossAtte = image_BG
        #     else:
        #         result_obj_NoCrossAtte = torch.cat([result_obj_NoCrossAtte, image_BG])
        #
        # #     image_show = self.image_processor.postprocess(image_BG, output_type="pil", do_denormalize=[True])
        # #
        # #     plt.subplot(2, BG_saved.shape[0], i + BG_saved.shape[0]+1)
        # #     plt.imshow(image_show[0])
        # #     plt.axis('off')
        # # plt.suptitle('seed = ' + str(seed) + ';  ' + 'prompt_bg = ' + prompt_bg[0] + ';  ' + 'prompt_obj = ' + prompt_obj[0],fontsize=40)
        # # pylab.show()
        # tensor_obj_NoCrossAtte = result_obj_NoCrossAtte
        # pil_obj_NoCrossAtte = self.image_processor.postprocess(result_obj_NoCrossAtte, output_type="pil", do_denormalize=[True]*BG_saved.shape[0])
        #
        # torch.cuda.empty_cache()


        return 1












