<h1 align="center">Video-as-Answer: Predict and Generate Next Video Event with Joint-GRPO</h1>

<div align='center'>

[Junhao Cheng<sup>1†</sup>](https://donahowe.github.io/),
[Liang Hou<sup>2</sup>](https://liang-hou.github.io/),
[Xin Tao<sup>2</sup>](https://www.xtao.website/),
[Jing Liao<sup>1</sup>](https://scholar.google.com/citations?user=3s9f9VIAAAAJ&hl=en)  
<sup>1</sup>City University of Hong Kong  <sup>2</sup>Kling Team, Kuaishou Technology  
<sup>†</sup> This work was conducted during the author's internship at Kling Team, Kuaishou Technology

<a href="https://video-as-answer.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-Video--as--Answer-blue.svg" height="20" />
</a>
<a href="http://arxiv.org/" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/Paper-Video--as--Answer-red?logo=arxiv" height="20" />
</a>
<a href="https://huggingface.co/" target="_blank">
    <img alt="HF Dataset: Video--as--Answer" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Video--as--Answer-ffc107?color=ffc107&logoColor=white" height="20" />
</a>

</div>

## 🔎 Introduction

<img src="assets/images/teaser.png" alt="Teaser Image" style="width: 100%; height: auto;">

We pioneer **Video-Next-Event Prediction (VNEP)**, extending text-based next-event prediction to dynamic video responses. This shift from *telling* to *showing* enables more intuitive and customized answers for procedural learning and creative exploration. 

To tackle VNEP, we propose **VANS**, a model that aligns a Vision-Language Model (VLM) with a Video Diffusion Model (VDM) through our **Joint-GRPO** post-training approach. Our method bridges the semantic-to-visual gap of VLM and VDM, enabling high-quality video event prediction and generation.

## 🏗️ Method

<div align="center">
  <table>
    <tr>
      <td align="center" width="35%">
        <img src="assets/images/model.png" alt="VANS Architecture" style="max-width: 100%; height: auto; max-height: 300px;">
        <br>
        <em>VANS Architecture: Dual-path processing with VLM for reasoning and VDM for generation</em>
      </td>
      <td align="center" width="45%">
        <img src="assets/images/grpo.png" alt="Joint-GRPO" style="max-width: 100%; height: auto; max-height: 300px;">
        <br>
        <em>Joint-GRPO: Two-stage co-steering optimization</em>
      </td>
    </tr>
  </table>
</div>

### Key Components

**VANS Architecture**: Processes input videos and questions through dual pathways:
- **VLM Path**: Performs instruction-grounded reasoning to generate textual captions
- **VDM Path**: Synthesizes videos conditioned on semantic captions and visual context

**Joint-GRPO**: Our two-stage reinforcement learning approach:
- **Stage 1**: Visualization-friendly VLM tuning - optimizes captions for visual plausibility
- **Stage 2**: Context-faithful VDM adaptation - ensures semantic alignment and visual coherence


## 🎯 Results

### 🍳 Procedural Teaching

<div align="center">

| Case | Input Video | Question | VANS Output |
|:----:|:-----------:|:--------:|:----------:|
| 1 | <video width="120" height="120" controls style="border-radius:6px;"><source src="assets/videos/proce_1/input.gif" type="video/mp4"></video> | "Show me the next step for baked chicken Parmesan." | <video width="120" height="120" controls style="border-radius:6px;"><source src="assets/videos/proce_1/jointgrpo.gif" type="video/mp4"></video> |
| 2 | <video width="120" height="120" controls style="border-radius:6px;"><source src="assets/videos/proce_2/input.gif" type="video/mp4"></video> | "Hi, I want to make slime. What should I do next?" | <video width="120" height="120" controls style="border-radius:6px;"><source src="assets/videos/proce_2/jointgrpo.gif" type="video/mp4"></video> |
| 3 | <video width="120" height="120" controls style="border-radius:6px;"><source src="assets/videos/proce_3/input.gif" type="video/mp4"></video> | "Hey AI assistant, I'm making a paper windmill and just uploaded a video. What should I do next?" | <video width="120" height="120" controls style="border-radius:6px;"><source src="assets/videos/proce_3/jointgrpo.gif" type="video/mp4"></video> |
| 4 | <video width="120" height="120" controls style="border-radius:6px;"><source src="assets/videos/proce_4/input.gif" type="video/mp4"></video> | "Hi, I'm making Samosa and just uploaded a video. What should I do next?" | <video width="120" height="120" controls style="border-radius:6px;"><source src="assets/videos/proce_4/jointgrpo.gif" type="video/mp4"></video> |
| 5 | <video width="120" height="120" controls style="border-radius:6px;"><source src="assets/videos/proce_5/input.gif" type="video/mp4"></video> | "My bike tire is flat. What should I do next?" | <video width="120" height="120" controls style="border-radius:6px;"><source src="assets/videos/proce_5/jointgrpo.gif" type="video/mp4"></video> |

</div>

---

### 🔮 Multi-Future Prediction

**Same input video, different questions lead to diverse future predictions:**

<div align="center">

#### 🎬 Scenario 1: Kitchen Reaction
<table>
<tr>
<td align="center" colspan="4"><strong>Input Video</strong></td>
</tr>
<tr>
<td align="center" colspan="4">
<video width="200" height="150" controls style="border-radius:8px;">
<source src="assets/videos/gen_1/input.gif" type="video/mp4">
</video>
</td>
</tr>
<tr>
<td align="center"><strong>Realistic</strong></td>
<td align="center"><strong>Dramatic</strong></td>
<td align="center"><strong>Comedic</strong></td>
</tr>
<tr>
<td align="center">
<video width="120" height="120" controls style="border-radius:6px;">
<source src="assets/videos/gen_1/con1.gif" type="video/mp4">
</video><br>
<small>"What is her reaction<br>if she gets burned?"</small>
</td>
<td align="center">
<video width="120" height="120" controls style="border-radius:6px;">
<source src="assets/videos/gen_1/con2.gif" type="video/mp4">
</video><br>
<small>"...in an exaggerated movie?"</small>
</td>
<td align="center">
<video width="120" height="120" controls style="border-radius:6px;">
<source src="assets/videos/gen_1/con3.gif" type="video/mp4">
</video><br>
<small>"...eats something spicy?"</small>
</td>
</tr>
</table>

#### 👵 Scenario 2: Emotional Responses
<table>
<tr>
<td align="center" colspan="3"><strong>Input Video</strong></td>
</tr>
<tr>
<td align="center" colspan="3">
<video width="200" height="150" controls style="border-radius:8px;">
<source src="assets/videos/gen_2/input.gif" type="video/mp4">
</video>
</td>
</tr>
<tr>
<td align="center">
<video width="120" height="120" controls style="border-radius:6px;">
<source src="assets/videos/gen_2/con1.gif" type="video/mp4">
</video><br>
<small>"Sees her grandson"</small>
</td>
<td align="center">
<video width="120" height="120" controls style="border-radius:6px;">
<source src="assets/videos/gen_2/con2.gif" type="video/mp4">
</video><br>
<small>"Sees her husband"</small>
</td>
<td align="center">
<video width="120" height="120" controls style="border-radius:6px;">
<source src="assets/videos/gen_2/con3.gif" type="video/mp4">
</video><br>
<small>"Sees death personification"</small>
</td>
</tr>
</table>

</div>




### 🔮 Multi-Future Prediction

**Same input video, different questions lead to diverse future predictions:**

<div align="center">
  <table>
    <tr>
      <td align="center" colspan="3"><strong>Input Video</strong></td>
    </tr>
    <tr>
      <td align="center" colspan="3">
        <video width="320" height="180" controls style="border-radius: 8px;">
          <source src="assets/videos/gen_1/input.gif" type="video/mp4">
        </video>
      </td>
    </tr>
  </table>

  <br>

  <table>
    <tr>
      <td align="center"><strong>Question 1</strong></td>
      <td align="center"><strong>Question 2</strong></td>
      <td align="center"><strong>Question 3</strong></td>
    </tr>
    <tr>
      <td align="center" width="33%">
        <em>"What is her reaction if she gets burned?"</em>
      </td>
      <td align="center" width="33%">
        <em>"What is her reaction if she gets burned in an exaggerated movie?"</em>
      </td>
      <td align="center" width="33%">
        <em>"What is her reaction if she eats something spicy in an exaggerated movie?"</em>
      </td>
    </tr>
    <tr>
      <td align="center">
        <video width="180" height="180" controls style="border-radius: 8px;">
          <source src="assets/videos/gen_1/con1.gif" type="video/mp4">
        </video>
      </td>
      <td align="center">
        <video width="180" height="180" controls style="border-radius: 8px;">
          <source src="assets/videos/gen_1/con2.gif" type="video/mp4">
        </video>
      </td>
      <td align="center">
        <video width="180" height="180" controls style="border-radius: 8px;">
          <source src="assets/videos/gen_1/con3.gif" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td align="center"><em>Realistic pain reaction</em></td>
      <td align="center"><em>Dramatic overreaction</em></td>
      <td align="center"><em>Comedic spicy reaction</em></td>
    </tr>
  </table>
</div>

**Key Insight:** VANS demonstrates strong conditional generation capabilities, producing contextually appropriate responses for different hypothetical scenarios.


## 🚀 Quick Start

### 🔮 Environment Setup


To set up the environment for inference, you can run the following command:
```shell
git clone https://github.com/KlingTeam/VANS.git
cd VANS
pip install requirements.txt
```

### 🌎 Download Models

To get started, download the VANS base models:
- **[Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)** - The Vision-Language Model
- **[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)** - The Video Diffusion Model

Then download the complete VANS model:  
**[VANS Model Download](https://huggingface.co/)** *(Coming Soon)*

### 🧸 Demo
To run local gradio demo:
```shell
python app.py
```

## 🚩 Plan
- [ ] Release Training Codes
- [ ] Release VANS-Data-100K dataset
- [ ] Release Inference Codes
- [ ] Release VANS model


## 📜 Citation

If you find our work helpful, please consider giving a star ⭐ and citation 📝

```BibTeXw

```
