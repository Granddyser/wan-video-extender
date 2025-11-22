# WAN Video Extender PRO - Memory Optimized

**Extend your WAN 2.2 videos with advanced features: per-loop prompts, LoRA switching, reference images, and smart overlap for maximum character consistency.**

**NEW:** Memory-optimized architecture with disk-based segment storage - efficient RAM usage even for long videos!

![Demo](demo.gif)

---

## 📦 Recommended Model

**Download:** [WAN2.2-14B-Rapid-AllInOne](https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne)

**Recommended Settings:**
- **Steps:** `4` (optimized for rapid generation)
- **Sampler:** `euler_a`
- **Scheduler:** `beta`
- **CFG:** `1.0`
- **Overlap Frames:** `16-24` (more = better consistency)

---

## 🚀 Features

### ✅ Core Features
- **Flexible Input Modes** - Video input (extend existing), Image input (animate static image), or No input (pure T2V generation)
- **Multi-Loop Video Extension** - Extend videos through multiple generation cycles
- **Memory Optimized** - Segment-based storage keeps RAM usage efficient
- **Smart Overlap System** - Use last N frames as context for seamless transitions (4-64 frames)
- **Custom Resolution** - Set any resolution when starting without input (256-2048px)

### 🎨 Advanced Character Consistency
- **Reference Image Support** - Use a character portrait for consistent appearance across all loops
- **Intermediate Image Injection** - Insert new images between loops to guide the generation
- **Per-Loop Prompts** - Different prompt for each loop (up to 10 loops)
- **LoRA Support** - Load different LoRAs for specific loops
- **Flexible Overlap** - Higher overlap = better consistency (recommended: 16-24 frames)

### 💾 Memory Optimization
- **Disk-Based Segments** - Each loop saves to disk, only overlap kept in RAM
- **Aggressive Cleanup** - Unloads all models between loops to prevent VRAM leaks
- **Smart Input Handling** - Efficient video processing
- **Final Assembly** - Combines all segments from disk at the end

---

## 🔧 Installation

1. **Copy the node file** to your ComfyUI custom nodes:
```bash
git clone https://github.com/Granddyser/wan-video-extender.git
```
```bash
pip install requirements.txt
```

---

## 🎯 Quick Start

### Input Modes

**1. Video Input (V2V)** - Extend existing video
```
Input: Video → WAN Video Extender PRO
Result: Original video + extended frames
```

**2. Image Input (I2V)** - Animate static image
```
Input: Image → WAN Video Extender PRO
Result: Animated video from still image
```

**3. No Input (T2V)** - Pure text-to-video generation
```
No input connected
Settings: default_width=832, default_height=480
Result: Video generated from prompt only
```

### Basic Settings

```
Settings:
  - extension_loops: 2-3
  - generate_frames: 81
  - overlap_frames: 16
  - steps: 4
  - cfg: 1.0
  - sampler: euler_a
  - scheduler: beta
  - positive_prompt: "your scene description"
```

---

## 🎓 Understanding Overlap

Overlap determines how many frames from your existing video are used as **context** for the next generation.

**How it works:**

```
Input: Your video frames

Loop 1:
  - Takes last N frames as context (overlap_frames)
  - Generates new frames
  - Adds only the new frames (skips overlap duplicates)
  
Loop 2:
  - Takes last N frames as context from previous output
  - Generates new frames
  - Adds new frames
  
Result: Original + all generated frames combined
```

### Overlap Settings Guide

| Overlap | New/Loop | Speed | Consistency | Best For |
|---------|----------|-------|-------------|----------|
| 8-12    | 69-73    | ⚡ Fast  | ⭐⭐        | Quick tests |
| 16-20   | 61-65    | 🚀 Good  | ⭐⭐⭐      | **Recommended** |
| 24-32   | 49-57    | ⏱️ OK    | ⭐⭐⭐⭐    | High quality |
| 40-48   | 33-41    | 🐌 Slow  | ⭐⭐⭐⭐⭐  | Maximum consistency |

**Rule of thumb:** More overlap = Better consistency, but slower generation.

---

## 🎨 Using Reference Images

Reference images drastically improve character consistency.

**Setup:**
```
LoadImage (Character portrait)
  ↓
WAN Video Extender PRO
  └─ reference_image input
```

**Tips:**
- Use a clear, high-quality portrait
- Alternative: First frame from your input video
- Combine with Character LoRA for best results
- Increase overlap to 24+ when using reference images

---

## 🔄 Per-Loop Prompts

Each loop can have its own prompt for storytelling.

**Example:**
```
extension_loops: 4

positive_prompt: "woman in forest"  (fallback)

prompt_loop_1: "woman walking through sunlit forest"
prompt_loop_2: "woman discovers ancient ruins"
prompt_loop_3: "woman examining mysterious artifact"
prompt_loop_4: "woman holding glowing artifact at sunset"
```

Result: A video with narrative progression!

---

## 🎭 Per-Loop LoRAs & Prompts

**Custom Prompts:**
Each loop can use a different prompt for scene changes.

**LoRA Support:**
Load different LoRAs for specific loops (without `.safetensors` extension).

---

## 💾 How Memory Optimization Works

**Our efficient approach:**
1. **Input**: Extract only overlap frames needed → Delete rest
2. **Generation**: Each loop generates segment
3. **Storage**: Save segment to disk (`/tmp/wan_segments_XXX/`)
4. **Memory**: Only overlap + current segment in RAM
5. **Final**: Load all segments from disk and combine

**Result:** Generate long videos without running out of RAM!

**Intermediate Images:** You can also inject new images between loops by connecting different images to the `image` input for each workflow run, allowing you to guide the generation at specific points.

---

## 🎯 Advanced Workflows

### Maximum Character Consistency
```
- Input: Portrait image
- reference_image: Same portrait
- extension_loops: 5
- overlap_frames: 24
- lora_loop_1: "your_character_lora" @ 0.8
- steps: 4, sampler: euler_a, scheduler: beta
```

### Long Video Generation
```
- Input: 81 frames
- extension_loops: 8
- overlap_frames: 20
- generate_frames: 81
- Result: Extended long-form video
```

### Style Evolution
```
- extension_loops: 4
- overlap_frames: 32
- Different LoRAs per loop for smooth style transitions
```

---


## 📝 Parameter Reference

### Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| extension_loops | 1 | 1-10 | How many times to extend |
| generate_frames | 81 | 16-200 | Frames per loop |
| overlap_frames | 16 | 4-64 | Context frames |
| steps | 20 | 1-100 | Sampling steps (try 4 for rapid) |
| cfg | 1.0 | 0-100 | CFG scale (1.0 recommended for WAN 2.2 Rapid) |
| sampler_name | - | - | euler_a recommended |
| scheduler | - | - | beta recommended |
| strength | 1.0 | 0-10 | VACE strength |
| seed | 0 | 0-∞ | Random seed |

### Optional Inputs

| Input | Description |
|-------|-------------|
| image | Input image to extend |
| video | Input video to extend |
| reference_image | Character reference for consistency |
| prompt_loop_1..10 | Per-loop custom prompts |
| lora_loop_1..10 | Per-loop LoRA files |
| lora_strength_1..10 | LoRA strengths |

---

## 📄 License

This node is provided as-is for use with ComfyUI and WAN 2.2.

---

## 🙏 Credits

- **WAN 2.2 Model** by the WAN team
- **ComfyUI** framework
- **Memory optimization** architecture

---

**Happy video generating! 🎬**
</document_content>
