import os
import csv
import shutil
import itertools
from PIL import Image
import torch

from vans import save_video, VideoData
from vans.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from vans.trainers.utils import VideoDataset

from config import (
    VDM_PRETRAINED_PATH, BASE_SAVE_PATH, CSV_FILE_PATH, 
    MLLM_PRETRAINED_PATH, TRAINED_CKPT_PATH, MLLM_TRAINED_CKPT_PATH
)


def setup_environment():
    """Setup CUDA environment and create directories"""
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)


def generate_prefixes():
    """Generate 5-character prefixes for file naming"""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    for chars in itertools.product(letters, repeat=5):
        yield ''.join(chars)


def initialize_pipeline():
    """Initialize the video generation pipeline"""
    model_configs = [
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern=f"{VDM_PRETRAINED_PATH}/diffusion_pytorch_model*.safetensors",
            offload_device="cpu"
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern=f"{VDM_PRETRAINED_PATH}/models_t5_umt5-xxl-enc-bf16.pth",
            offload_device="cpu"
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", 
            origin_file_pattern=f"{VDM_PRETRAINED_PATH}/Wan2.1_VAE.pth",
            offload_device="cpu"
        ),
    ]
    
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        mllm_pretrained_path=MLLM_PRETRAINED_PATH,
        use_mllm=True,
        trained_ckpt_path=TRAINED_CKPT_PATH,
        mllm_trained_ckpt_path=MLLM_TRAINED_CKPT_PATH
    )


def process_video_row(pipe, dataset, row, prefix):
    """Process a single row from CSV and generate video"""
    input_video_path = row.get('input_video_path', '')
    output_video_path = row.get('\ufeffoutput_video_path', '')
    eng_gt_caption = row.get('ENG_GT_Caption', '')
    instructions = row.get('ENG_Instruction', '')

    # Load reference video
    ref_VAE_video = dataset.load_video_imageio(input_video_path)
    
    # Generate video
    text, video = pipe(
        prompt=eng_gt_caption,
        instructions=instructions,
        input_video_path=input_video_path,
        negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, stationary, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, stationary frame, cluttered background, three legs, crowded background people, walking backwards",
        seed=0,
        tiled=True,
        height=352,
        width=640,
        num_frames=33,
        ref_VAE_video=ref_VAE_video,
    )
    
    # Save generated video
    save_path = os.path.join(BASE_SAVE_PATH, f"{prefix}_gen.mp4")
    save_video(video, save_path, fps=11, quality=5)
    
    # Optionally copy input and output videos
    # in_save_path = os.path.join(BASE_SAVE_PATH, f"{prefix}_in.mp4")
    # out_save_path = os.path.join(BASE_SAVE_PATH, f"{prefix}_out.mp4")
    # shutil.copy(input_video_path, in_save_path)
    # shutil.copy(output_video_path, out_save_path)


def main():
    """Main execution function"""
    setup_environment()
    
    # Initialize components
    dataset = VideoDataset(
        height=352,
        width=640, 
        num_frames=33,
    )
    
    pipe = initialize_pipeline()
    prefix_gen = generate_prefixes()
    
    # Process CSV file
    with open(CSV_FILE_PATH, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row_number, row in enumerate(csv_reader, 1):
            prefix = next(prefix_gen)
            print(f"Processing row {row_number} with prefix: {prefix}")
            
            try:
                process_video_row(pipe, dataset, row, prefix)
                print(f"Successfully processed row {row_number}")
            except Exception as e:
                print(f"Error processing row {row_number}: {e}")
                continue

if __name__ == "__main__":
    main()