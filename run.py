import gc
import os
import time
import psutil
import numpy as np
import torch
from torch.nn import DataParallel
import logging # Added
from typing import Optional # Added

from diffusers.training_utils import set_seed
from diffusers import AutoencoderKLTemporalDecoder
from fire import Fire

from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
from normalcrafter.utils import vis_sequence_normal, save_video, read_video_frames


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
        logger=None, # Added
        use_nvtx: bool = False, # Added
        use_tensorrt: bool = False, # Added
    ):
        self.logger = logger if logger else logging.getLogger(__name__) # Added
        self.use_nvtx = use_nvtx # Added
        self.use_tensorrt = use_tensorrt # Added
        self.cpu_offload = cpu_offload # Added to store for later check

        unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
            unet_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            unet_path, subfolder="vae"
        )
        weight_dtype = torch.float16
        vae.to(dtype=weight_dtype)
        unet.to(dtype=weight_dtype)
        # load weights of other components from the provided checkpoint
        self.pipe = NormalCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            vae=vae,
            torch_dtype=weight_dtype,
            variant="fp16",
            logger=self.logger, # Added
            use_nvtx=self.use_nvtx, # Added
        )

        if self.use_tensorrt and self.cpu_offload is None and torch.cuda.is_available():
            self.logger.info("Attempting TensorRT compilation...")
            # Ensure components are on CUDA before TensorRT compilation
            if not next(self.pipe.unet.parameters()).is_cuda:
                self.pipe.unet.to("cuda")
            if not next(self.pipe.vae.parameters()).is_cuda:
                self.pipe.vae.to("cuda")

            try:
                self.logger.info("Attempting TensorRT compilation for UNet...")
                self.pipe.unet = torch.compile(
                    self.pipe.unet,
                    backend="torch_tensorrt",
                    options={
                        "enabled_precisions": {torch.float16},
                        "truncate_long_and_double": True,
                    },
                    dynamic=False
                )
                self.logger.info("UNet compiled with TensorRT successfully.")
            except Exception as e:
                self.logger.error(f"TensorRT compilation for UNet failed: {e}. Proceeding without TensorRT for UNet.")

            try:
                self.logger.info("Attempting TensorRT compilation for VAE...")
                self.pipe.vae = torch.compile(
                    self.pipe.vae,
                    backend="torch_tensorrt",
                    options={
                        "enabled_precisions": {torch.float16},
                        "truncate_long_and_double": True,
                    },
                    dynamic=False
                )
                self.logger.info("VAE compiled with TensorRT successfully.")
            except Exception as e:
                self.logger.error(f"TensorRT compilation for VAE failed: {e}. Proceeding without TensorRT for VAE.")

        if self.cpu_offload is None and torch.cuda.device_count() > 1: # Modified to use self.cpu_offload
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for UNet (DataParallel).") # Modified
            self.pipe.unet = DataParallel(self.pipe.unet)

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if self.cpu_offload is not None: # Modified to use self.cpu_offload
            if self.cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif self.cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {self.cpu_offload}")
        else:
            self.pipe.to("cuda")
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            self.logger.warning(e) # Modified
            self.logger.warning("Xformers is not enabled") # Modified
        # self.pipe.enable_attention_slicing()

    def infer(
        self,
        video: str,
        save_folder: str = "./demo_output",
        window_size: int = 14,
        time_step_size: int = 10,
        process_length: int = 195,
        decode_chunk_size: int = 7,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = 15,
        seed: int = 42,
        save_npz: bool = False,
    ):
        set_seed(seed)

        self.logger.info(f"Initial RAM used: {psutil.virtual_memory().used / (1024**3):.2f} GB") # Modified
        if torch.cuda.is_available():
            self.logger.info(f"Initial VRAM used (GPU 0): {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB") # Modified

        frames, target_fps = read_video_frames(
            video,
            process_length,
            target_fps,
            max_res,
        )
        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            if self.use_nvtx and torch.cuda.is_available(): # Added
                torch.cuda.nvtx.range_push("NormalCrafterPipeline.call") # Added
            start_time = time.time()
            res = self.pipe(
                frames,
                decode_chunk_size=decode_chunk_size,
                time_step_size=time_step_size,
                window_size=window_size,
            ).frames[0]
            end_time = time.time()
            if self.use_nvtx and torch.cuda.is_available(): # Added
                torch.cuda.nvtx.range_pop() # Added
            self.logger.info(f"Pipeline processing time: {end_time - start_time:.2f} seconds") # Modified
            self.logger.info(f"RAM used after pipeline: {psutil.virtual_memory().used / (1024**3):.2f} GB") # Modified
            if torch.cuda.is_available():
                self.logger.info(f"VRAM used after pipeline (GPU 0): {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB") # Modified
        # visualize the depth map and save the results
        vis = vis_sequence_normal(res)
        # save the depth map and visualization with the target FPS
        save_path = os.path.join(
            save_folder, os.path.splitext(os.path.basename(video))[0]
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_video(vis, save_path + "_vis.mp4", fps=target_fps)
        save_video(frames, save_path + "_input.mp4", fps=target_fps)
        if save_npz:
            np.savez_compressed(save_path + ".npz", depth=res)

        return [
            save_path + "_input.mp4",
            save_path + "_vis.mp4",
        ]

    def run(
        self,
        input_video,
        num_denoising_steps,
        guidance_scale,
        max_res=1024,
        process_length=195,
    ):
        res_path = self.infer(
            input_video,
            num_denoising_steps,
            guidance_scale,
            max_res=max_res,
            process_length=process_length,
        )
        # clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()
        return res_path[:2]


def main(
    video_path: str,
    save_folder: str = "./demo_output",
    unet_path: str = "Yanrui95/NormalCrafter",
    pre_train_path: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    process_length: int = -1,
    cpu_offload: str = "model",
    target_fps: int = -1,
    seed: int = 42,
    window_size: int = 14,
    time_step_size: int = 10,
    max_res: int = 1024,
    dataset: str = "open",
    save_npz: bool = False,
    log_file: Optional[str] = None, # Added
    use_nvtx: bool = False, # Added
    use_tensorrt: bool = False, # Added
):
    # Setup Logger
    logger = logging.getLogger(__name__) # Added
    logger.setLevel(logging.INFO) # Added
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Added

    ch = logging.StreamHandler() # Added
    ch.setFormatter(formatter) # Added
    logger.addHandler(ch) # Added

    if log_file: # Added
        fh = logging.FileHandler(log_file) # Added
        fh.setFormatter(formatter) # Added
        logger.addHandler(fh) # Added

    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_train_path=pre_train_path,
        cpu_offload=cpu_offload,
        logger=logger, # Added
        use_nvtx=use_nvtx, # Added
        use_tensorrt=use_tensorrt, # Added
    )
    # process the videos, the video paths are separated by comma
    video_paths = video_path.split(",")
    for video in video_paths:
        depthcrafter_demo.infer(
            video,
            save_folder=save_folder,
            window_size=window_size,
            process_length=process_length,
            time_step_size=time_step_size,
            max_res=max_res,
            dataset=dataset,
            target_fps=target_fps,
            seed=seed,
            save_npz=save_npz,
            decode_chunk_size=decode_chunk_size,
        )
        # clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # running configs
    # the most important arguments for memory saving are `cpu_offload`, `enable_xformers`, `max_res`, and `window_size`
    # the most important arguments for trade-off between quality and speed are
    # `num_inference_steps`, `guidance_scale`, and `max_res`
    Fire(main)
