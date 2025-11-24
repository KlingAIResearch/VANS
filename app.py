import os
# ⚠️ Keep Gradio server timeout increased to 600 seconds (10 minutes)
os.environ['GRADIO_SERVER_TIMEOUT'] = '600'  # 10 minutes
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import gradio as gr
import torch
import time
import json
from pathlib import Path
import threading
import signal
import psutil
import os

# Import necessary tools from vans library
from vans import save_video
# Ensure vans library and related dependencies are properly installed
from vans.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from vans.trainers.utils import VideoDataset 

# Import configuration
try:
    from config import (
        VDM_PRETRAINED_PATH, MLLM_PRETRAINED_PATH, 
        TRAINED_CKPT_PATH, MLLM_TRAINED_CKPT_PATH
    )
except ImportError:
    print("Warning: config.py not found. Using dummy paths.")
    # Please replace these paths with your actual model storage paths!
    VDM_PRETRAINED_PATH = "./models/vdm"
    MLLM_PRETRAINED_PATH = "./models/mllm"
    TRAINED_CKPT_PATH = "./ckpts/trained"
    MLLM_TRAINED_CKPT_PATH = "./ckpts/mllm_trained"

# Set up decord bridge
import decord
decord.bridge.set_bridge("torch")

# --- Global model initialization ---
print("Initializing VANS model...")
pipe = None
try:
    # Ensure dataset object can be correctly initialized
    dataset = VideoDataset(height=352, width=640, num_frames=33)

    # Initialize model configs
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

    # Initialize pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda", # Ensure running on machine with CUDA device
        model_configs=model_configs,
        mllm_pretrained_path=MLLM_PRETRAINED_PATH,
        use_mllm=True,
        trained_ckpt_path=TRAINED_CKPT_PATH,
        mllm_trained_ckpt_path=MLLM_TRAINED_CKPT_PATH
    )
    print("VANS model initialized successfully!")
except Exception as e:
    print(f"Error during model initialization: {e}")
    import traceback
    traceback.print_exc()
    pipe = None

class ProcessController:
    def __init__(self):
        self.current_process = None
        self.should_stop = False
        self.lock = threading.Lock()
    
    def set_current_process(self, process_info):
        with self.lock:
            self.current_process = process_info
            self.should_stop = False
    
    def stop_current_process(self):
        with self.lock:
            self.should_stop = True
            if self.current_process:
                print("Attempting to stop current generation process...")
                try:
                    if hasattr(self.current_process, 'cancel'):
                        self.current_process.cancel()
                    
                    current_pid = os.getpid()
                    parent = psutil.Process(current_pid)
                    children = parent.children(recursive=True)
                    
                    for child in children:
                        try:
                            child.terminate()
                        except:
                            pass
                    
                    time.sleep(1)
                    
                    children = parent.children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except:
                            pass
                            
                    print("Generation process stopped")
                except Exception as e:
                    print(f"Error stopping process: {e}")
                finally:
                    self.current_process = None

process_controller = ProcessController()

# --- Status Management ---
class GenerationStatus:
    def __init__(self):
        self.current_generation = None
        self.status_file = Path("./demo_outputs/generation_status.json")
        # Ensure demo_outputs directory exists
        self.status_file.parent.mkdir(exist_ok=True)
    
    def set_generating(self, video_path, question):
        self.current_generation = {
            "video_path": video_path,
            "question": question,
            "start_time": time.time(),
            "status": "generating"
        }
        self._save_status()
    
    def set_completed(self, result_path, text_response):
        if self.current_generation:
            self.current_generation.update({
                "status": "completed",
                "result_path": result_path,
                "text_response": text_response,
                "completion_time": time.time()
            })
            self._save_status()
    
    def set_failed(self, error_msg):
        if self.current_generation:
            self.current_generation.update({
                "status": "failed",
                "error": error_msg,
                "completion_time": time.time()
            })
            self._save_status()
    
    def set_interrupted(self):
        if self.current_generation:
            self.current_generation.update({
                "status": "interrupted",
                "completion_time": time.time()
            })
            self._save_status()
    
    def get_status(self):
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def _save_status(self):
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.current_generation, f)
        except Exception as e:
            print(f"Error saving status: {e}")

status_manager = GenerationStatus()

# --- Example loading function ---
def load_demo_examples():
    """Load demo examples from the demo folders"""
    examples = []
    demo_folders = {
        "Predictive": "./demo/predictive",
        "Procedural": "./demo/procedural"
    }
    
    for category, folder_path in demo_folders.items():
        videos_dir = Path(folder_path) / "videos"
        questions_dir = Path(folder_path) / "questions"
        
        if videos_dir.exists() and questions_dir.exists():
            for i in range(1, 5): 
                video_path = videos_dir / f"{i}.mp4"
                question_path = questions_dir / f"{i}.txt"
                
                if video_path.exists() and question_path.exists():
                    try:
                        with open(question_path, 'r', encoding='utf-8') as f:
                            question = f.read().strip()
                        
                        examples.append({
                            "category": category,
                            "video_path": str(video_path), 
                            "question": question,
                            "name": f"{category} Example {i}"
                        })
                    except Exception as e:
                        print(f"Error loading demo example {i} in {category}: {e}")
                        
    return examples

demo_examples_data = load_demo_examples()

def should_stop_generation():
    return process_controller.should_stop

def generate_video_response(input_video, question, progress=gr.Progress()):
    """Generate video response using VANS model"""
    
    if pipe is None:
        return "Error: Model initialization failed. Please check the console.", None
    
    process_controller.set_current_process({
        "input_video": input_video,
        "question": question,
        "start_time": time.time()
    })
    
    temp_video_path = None
    
    try:
        if input_video is None:
            return "Please upload a video file", None
        
        if should_stop_generation():
            return "Generation was stopped by user", None
        
        status_manager.set_generating(input_video, question)
        
        progress(0.05, desc="Starting process...")
        print(f"Processing video: {input_video}")
        print(f"Question: {question}")
        
        if should_stop_generation():
            return "Generation was stopped by user", None
        
        temp_dir = Path("./demo_outputs")
        temp_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        temp_video_path = temp_dir / f"generated_{timestamp}.mp4" 
        
        progress(0.1, desc="Loading reference video...")
        
        if should_stop_generation():
            return "Generation was stopped by user", None
            
        ref_VAE_video = dataset.load_video_imageio(input_video)
        
        progress(0.3, desc="Generating response (This may take a while)...")
        print("Starting video generation...")
        
        if should_stop_generation():
            return "Generation was stopped by user", None
        
        # --- Core generation call ---
        text_response, generated_video = pipe(
            prompt=question,
            instructions=question,
            input_video_path=input_video,  
            negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, stationary, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, stationary frame, cluttered background, three legs, crowded background people, walking backwards",
            seed=0,
            tiled=True,
            height=352,
            width=640,
            num_frames=33,
            ref_VAE_video=ref_VAE_video,
        )
        # --- End core generation call ---
        
        if should_stop_generation():
            return "Generation was stopped by user", None
        
        progress(0.9, desc="Saving generated video...")
        print("Video generation completed, saving video...")
        
        # Save generated video
        save_video(generated_video, str(temp_video_path), fps=11, quality=5)
        
        if should_stop_generation():
            return "Generation was stopped by user", None
        
        # Verify file and return path
        if temp_video_path.exists() and temp_video_path.stat().st_size > 0:
            # Command line output success message
            print(f"Video successfully saved to: {temp_video_path}")
            
            output_path = str(temp_video_path) 
            
            # Set completion status
            status_manager.set_completed(output_path, text_response)
            
            time.sleep(1)  # Ensure file is completely written
            
            progress(1.0, desc="Done!")
            # Command line output return information
            print(f"Returning text response and video path: {output_path}")
            
            process_controller.set_current_process(None)
            
            # Key return: return path, Gradio will automatically display it to video_output
            return text_response, output_path
        else:
            error_msg = "Error: Generated video file is empty or failed to save."
            print(error_msg)
            status_manager.set_failed(error_msg)
            progress(1.0, desc="Failed")
            
            process_controller.set_current_process(None)
            
            return error_msg, None
            
    except Exception as e:
        if should_stop_generation():
            status_manager.set_interrupted()
            process_controller.set_current_process(None)
            return "Generation was stopped by user", None
            
        error_msg = f"Error generating video: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        status_manager.set_failed(error_msg)
        progress(1.0, desc="Failed")
        
        process_controller.set_current_process(None)
        
        return error_msg, None

# --- Recovery functions ---
def clear_outputs_only():
    """Clear only the output components while keeping inputs unchanged"""
    print("Clearing outputs only...")
    return "", None

def stop_and_recover():
    print("=== FORCE STOPPING AND RECOVERING ===")
    
    process_controller.stop_current_process()
    
    status_manager.set_interrupted()
    
    time.sleep(0.5)
    
    return recover_last_generation()

def recover_last_generation():
    """Recover last generation results after clearing outputs"""
    print("=== RECOVERING LAST GENERATION ===")
    
    # Read status directly from file
    status_file = Path("./demo_outputs/generation_status.json")
    if not status_file.exists():
        print("No status file found")
        return "No generation history found.", None
    
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        text_response = status.get("text_response", "")
        result_path = status.get("result_path", "")
        
        print(f"Text response: {text_response}")
        print(f"Result path: {result_path}")
        
        # Check if video file exists and is valid
        video_output_value = None
        if result_path and Path(result_path).exists() and Path(result_path).is_file():
            file_size = Path(result_path).stat().st_size
            print(f"Video file exists, size: {file_size} bytes")
            if file_size > 0:
                print("Returning valid video and text")
                video_output_value = result_path
        
        # If no valid video but we have text
        if not text_response:
            text_response = "No text response available"
            
        print(f"Final return - Text: {text_response}, Video: {video_output_value}")
        return text_response, video_output_value
            
    except Exception as e:
        print(f"Error reading status: {e}")
        import traceback
        traceback.print_exc()
        return f"Error recovering: {str(e)}", None

# --- Gradio interface creation function ---
def create_demo():
    """Create the Gradio demo interface"""
    
    # Enable queue in Blocks
    with gr.Blocks(
        theme=gr.themes.Soft(), 
        title="VANS Demo",
        analytics_enabled=False
    ) as demo: 
        
        gr.Markdown(
            """
            # 🎬 VANS Demo
            ### Video-as-Answer: Predict and Generate Next Video Event with Joint-GRPO
            
            Upload a video and ask a question about what happens next. The model will generate both text and video answers.
            
            **If generation is stuck or you want to see previous results, click the yellow RECOVER button below!**
            """
        )
        
        # Hidden status component
        status_component = gr.JSON(visible=False, value={"status": "idle"})
        
        # Recovery notice
        gr.Markdown(
            """
            <div style="padding: 10px; background: #fff3cd; border-radius: 5px; margin-bottom: 15px; color: #000000;">
            💡 IMPORTANT: If generation is taking too long or you don't see results, click the YELLOW "SHOW LAST RESULTS" button below to immediately stop current process and show last results!
            </div>
            """,
            elem_id="recovery-notice"
        )
        
        # Main Input/Output Section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                
                with gr.Group():
                    input_video = gr.Video(
                        label="Input Video",
                        sources=["upload"],
                        height=300,
                        format="mp4"
                    )
                    
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="Ask what happens next in the video... (e.g., 'What will happen next?', 'How will this action continue?')",
                        lines=3
                    )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                
                with gr.Group():
                    with gr.Tab("Text Response"):
                        text_output = gr.Textbox(
                            label="Text Answer",
                            lines=4,
                            interactive=False,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("Generated Video"):
                        video_output = gr.Video(
                            label="Generated Video Answer",
                            height=300,
                            interactive=False,
                            format="mp4",
                            autoplay=True,
                            show_share_button=True
                        )
        
        # Action Buttons
        with gr.Row():
            with gr.Column(scale=1):
                clear_btn = gr.Button("🗑️ Clear Outputs", variant="secondary", size="lg")
            
            with gr.Column(scale=2):
                generate_btn = gr.Button(
                    "🎬 Generate Video Answer", 
                    variant="primary",
                    size="lg"
                )
        
        # Recovery button with yellow styling - BIG and PROMINENT
        with gr.Row():
            recover_btn = gr.Button(
                "🔄 SHOW LAST RESULT", 
                variant="secondary", 
                size="lg",
                elem_classes="recover-button"
            )
        
        # Quick Examples Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 Quick Examples")
                gr.Markdown("Click any example to load video and question, then click the **Generate Video Answer** button above")
                
                categories = {}
                for example in demo_examples_data:
                    if example["category"] not in categories:
                        categories[example["category"]] = []
                    categories[example["category"]].append(example)
                
                for category, examples_list in categories.items():
                    gr.Markdown(f"#### {category} Examples")
                    
                    with gr.Row():
                        for example in examples_list:
                            with gr.Column(scale=1, min_width=200):
                                with gr.Group():
                                    gr.Video(
                                        value=example["video_path"],
                                        label=example["name"],
                                        interactive=False,
                                        height=150,
                                        show_download_button=False,
                                        format="mp4"
                                    )

                                    gr.Textbox(
                                        value=example["question"],
                                        label="Question",
                                        interactive=False,
                                        lines=2,
                                        max_lines=2
                                    )

                                    load_btn = gr.Button(
                                        f"Load {example['name'].split()[-1]}", 
                                        size="sm",
                                        variant="secondary"
                                    )
                                
                                    # Load button also bypasses queue
                                    load_btn.click(
                                        lambda v_path=example["video_path"], q_text=example["question"]: [v_path, q_text, "", None],
                                        outputs=[input_video, question_input, text_output, video_output],
                                        queue=False
                                    )
        
        # Info Section
        with gr.Row():
            with gr.Column():
                with gr.Accordion("ℹ️ About VANS", open=False):
                    gr.Markdown("""
                    **VANS (Video-as-Answer)** is a novel approach that:
                    - Predicts and generates the next video event based on input video and question
                    - Uses Joint-GRPO training for improved video generation
                    - Provides both textual and visual answers to video-based questions
                    
                    **Recovery Feature**: Click the YELLOW "Force Stop & Recover" button anytime to immediately STOP current generation and show your last completed results!
                    """)

        # Bind events
        generate_btn.click(
            fn=generate_video_response,
            inputs=[input_video, question_input],
            outputs=[text_output, video_output],
            api_name="generate_video",
            queue=True,
            show_progress="full"
        )

        recover_btn.click(
            fn=stop_and_recover,
            inputs=[],
            outputs=[text_output, video_output],
            queue=False
        )
        
        clear_btn.click(
            fn=lambda: [None, "", "", None],
            inputs=[],
            outputs=[input_video, question_input, text_output, video_output],
            queue=False
        )
        
        # Add custom CSS for recover button - make it very prominent
        demo.css = """
        .recover-button {
            background: linear-gradient(145deg, #ff6b6b, #ee5a24) !important;
            border-color: #ff6b6b !important;
            color: #ffffff !important;
            font-weight: bold !important;
            font-size: 16px !important;
            padding: 15px 30px !important;
            border-width: 3px !important;
        }
        .recover-button:hover {
            background: linear-gradient(145deg, #ee5a24, #d63031) !important;
            border-color: #ffffff !important;
            transform: scale(1.05) !important;
            color: #ffffff !important;
        }
        """
        
    return demo

if __name__ == "__main__":
    demo = create_demo()
    print(f"Gradio Server Timeout set to {os.environ.get('GRADIO_SERVER_TIMEOUT', 'Default')}")
    print("Launching Gradio Demo...")
    
    # Keep running parameters unchanged
    demo.launch(
        server_name="0.0.0.0",
        server_port=26006,
        share=False,
        debug=True,
        max_file_size="100MB",
        inbrowser=False,
        ssl_verify=False
    )
