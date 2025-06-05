import PIL
from PIL import Image
import requests
import torch
from io import BytesIO
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pylab
from diffusers import StableDiffusionInpaintPipeline,ControlNetModel,StableDiffusionControlNetPipeline,StableDiffusionControlNetPipeline,DPMSolverMultistepScheduler
from pipeline_video_SD import SDInpaintVideoPipeline
import numpy as np
import math
import utils
from diffusers.image_processor import VaeImageProcessor

import os
from pipeline_video_SD_bycall import SDInpaintVideoPipeline_bycall

from utlis.orf_find import compute_optimal_reference_frame,load_pose_sequence_from_folder

from pipeline_video_SD_add_controlnet import controlnet_inpaint_video_pipeline
from maskingapp import MaskingApp
from CurveDrawingApp import CurveDrawingApp
import argparse
from omegaconf import OmegaConf
from RectangleDrawerApp import RectangleDrawerApp


def main(
    pretrained_ControlNet_Model_path: str,
    pretrained_inpaint_Model_path: str,
    save_path: str,
    prompt_obj: list,
    prompt_bg: list,
    negative_prompt_bg: list,
    negative_prompt_obj: list,
    seed_bg: int,
    seed_obj: int,
    frame_num: int,
    crop_size: int,
    pose_image_path: str = None,
    org_bg_image_path: str = '',
    config_file: str = ''
):
    device = "cuda"

    controlnet = ControlNetModel.from_pretrained(
        pretrained_ControlNet_Model_path,
        torch_dtype=torch.float16
    )

    pipe = controlnet_inpaint_video_pipeline.from_pretrained(
        pretrained_inpaint_Model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if org_bg_image_path == '':
        bg_image = None
        print("Background-free generation mode")
    elif os.path.isdir(org_bg_image_path):
        bg_image = []
        for filename in os.listdir(org_bg_image_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = PIL.Image.open(os.path.join(org_bg_image_path, filename)).convert('RGB')
                img = img.resize((512,512))
                bg_image.append(img)
                # plt.imshow(img)
                # pylab.show()
        print('Multi-background generation mode')
    elif os.path.isfile(org_bg_image_path):
        bg_image = PIL.Image.open(org_bg_image_path).convert('RGB').resize((512, 512))
        print('Single-background generation mode')
    else:
        print(f"{org_bg_image_path} Invalid path")

    prompt_obj = list(prompt_obj)
    prompt_bg = list(prompt_bg)

    negative_prompt_bg = list(negative_prompt_bg)
    negative_prompt_obj = list(negative_prompt_obj)
    pipe(
        prompt_bg=prompt_bg,
        prompt_obj=prompt_obj,
        frame_num=frame_num,
        save_path=save_path,
        image=bg_image,
        image_pose_paths=pose_image_path,
        seed_bg=seed_bg,
        seed_obj=seed_obj,
        Maskingapp=MaskingApp,
        CurveDrawingApp=CurveDrawingApp,
        RectangleDrawerApp = RectangleDrawerApp,
        crop_size=crop_size,
        negative_prompt_bg=negative_prompt_bg,
        negative_prompt_obj=negative_prompt_obj,
        config_file=config_file,
        device="cuda"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./configs/Track.yaml")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    conf['config_file'] = parser.get_default('config')
    main(**conf)




