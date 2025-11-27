import torch
import numpy as np
import comfy.model_management
import comfy.sample
import comfy.utils
import comfy.sd
import comfy.latent_formats
import nodes
import node_helpers
import folder_paths
import cv2
import os
import copy
import tempfile
import shutil
import gc


def get_lora_list():
    """Get list of available LoRAs with 'None' as first option.
    Always returns at least ['None'] even if no LoRAs are installed.
    """
    try:
        loras = folder_paths.get_filename_list("loras")
        if loras is None:
            loras = []
    except Exception:
        loras = []
    return ["None"] + list(loras)


class WanVideoExtenderLowHigh:
    @classmethod
    def INPUT_TYPES(s):
        lora_list = get_lora_list()
        return {
            "required": {
                "high_noise": ("MODEL",),
                "low_noise": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),

                # MAIN PROMPTS
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": " ",
                    "placeholder": "Main prompt for a single run, image-to-video, or video extension input. If using more than two connections, select the loop input for the string and leave this field empty. Keep it short, this is only a basic template.",
                    "tooltip": "Main-Prompt / Basis-Prompt für alle Loops (Fallback)."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "low quality, blurry, glitch, distortion"
                }),

                # HIGH NOISE SAMPLER
                "sampler_name_high": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_high": (comfy.samplers.KSampler.SCHEDULERS,),
                "steps_h": ("INT", {"default": 4, "min": 1, "max": 100}),
                "start_at_step_h": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step_h": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "cfg_h": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),

                # LOW NOISE SAMPLER
                "sampler_name_low": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_low": (comfy.samplers.KSampler.SCHEDULERS,),
                "steps_l": ("INT", {"default": 4, "min": 1, "max": 100}),
                "start_at_step_l": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step_l": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "cfg_l": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),

                # EXTENSION SETTINGS
                "extension_loops": ("INT", {"default": 1, "min": 1, "max": 10}),
                "generate_frames": ("INT", {"default": 81, "min": 16, "max": 200}),
                "overlap_frames": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "tooltip": "Anzahl Überlappungsframes zwischen Loops (Kontext)."
                }),
                "empty_frame_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                # Resolution (nur wenn kein Input)
                "default_width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "default_height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 8}),
            },
            "optional": {
                # GLOBAL INPUT
                "image": ("IMAGE",),
                "video": ("*",),
                "reference_image": ("IMAGE", {"tooltip": "Globales Character-Referenzbild für VACE"}),
                "inpaint_mask": ("MASK",),

                # LOOP 1: Prompt + LoRA HIGH + LoRA LOW + Image
                "prompt_loop_1": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Optionaler Loop-1-Prompt (String-Node anschließen, leer = Base Prompt)."
                }),
                "lora_high_loop_1": (lora_list, {"tooltip": "LoRA für HIGH noise model Loop 1 (None = keine LoRA)"}),
                "lora_high_strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low_loop_1": (lora_list, {"tooltip": "LoRA für LOW noise model Loop 1 (None = keine LoRA)"}),
                "lora_low_strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_loop_1": ("IMAGE", {
                    "tooltip": "Optionales Bild / Bilder für Loop 1. Wenn gesetzt, harter Schnitt: diese Frames als Kontext statt vorheriger 16 Frames."
                }),
                "use_reference_loop_1": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 1"}),
                "use_overlap_loop_1": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 1"}),
                "use_endframe_loop_1": ("BOOLEAN", {"default": False, "tooltip": "Nutze Bild(er) des nächsten Loops als Endframe für Loop 1 (VACE Endframe)."}),
                "reference_image_1": ("IMAGE", {"tooltip": "Character-Referenz nur für Loop 1"}),

                # LOOP 2
                "prompt_loop_2": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 2 Prompt (leer = Base Prompt)"}),
                "lora_high_loop_2": (lora_list, {"tooltip": "LoRA für HIGH noise model Loop 2 (None = keine LoRA)"}),
                "lora_high_strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low_loop_2": (lora_list, {"tooltip": "LoRA für LOW noise model Loop 2 (None = keine LoRA)"}),
                "lora_low_strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_loop_2": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 2 (harte Schnitt-Quelle)."}),
                "use_reference_loop_2": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 2"}),
                "use_overlap_loop_2": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 2"}),
                "use_endframe_loop_2": ("BOOLEAN", {"default": False, "tooltip": "Nutze Bild(er) des nächsten Loops als Endframe für Loop 2 (VACE Endframe)."}),
                "reference_image_2": ("IMAGE", {"tooltip": "Character-Referenz nur für Loop 2"}),

                # LOOP 3
                "prompt_loop_3": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 3 Prompt (leer = Base Prompt)"}),
                "lora_high_loop_3": (lora_list, {"tooltip": "LoRA für HIGH noise model Loop 3 (None = keine LoRA)"}),
                "lora_high_strength_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low_loop_3": (lora_list, {"tooltip": "LoRA für LOW noise model Loop 3 (None = keine LoRA)"}),
                "lora_low_strength_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_loop_3": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 3 (harte Schnitt-Quelle)."}),
                "use_reference_loop_3": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 3"}),
                "use_overlap_loop_3": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 3"}),
                "use_endframe_loop_3": ("BOOLEAN", {"default": False, "tooltip": "Nutze Bild(er) des nächsten Loops als Endframe für Loop 3 (VACE Endframe)."}),
                "reference_image_3": ("IMAGE", {"tooltip": "Character-Referenz nur für Loop 3"}),

                # LOOP 4
                "prompt_loop_4": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 4 Prompt (leer = Base Prompt)"}),
                "lora_high_loop_4": (lora_list, {"tooltip": "LoRA für HIGH noise model Loop 4 (None = keine LoRA)"}),
                "lora_high_strength_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low_loop_4": (lora_list, {"tooltip": "LoRA für LOW noise model Loop 4 (None = keine LoRA)"}),
                "lora_low_strength_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_loop_4": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 4 (harte Schnitt-Quelle)."}),
                "use_reference_loop_4": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 4"}),
                "use_overlap_loop_4": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 4"}),
                "use_endframe_loop_4": ("BOOLEAN", {"default": False, "tooltip": "Nutze Bild(er) des nächsten Loops als Endframe für Loop 4 (VACE Endframe)."}),
                "reference_image_4": ("IMAGE", {"tooltip": "Character-Referenz nur für Loop 4"}),

                # LOOP 5
                "prompt_loop_5": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 5 Prompt (leer = Base Prompt)"}),
                "lora_high_loop_5": (lora_list, {"tooltip": "LoRA für HIGH noise model Loop 5 (None = keine LoRA)"}),
                "lora_high_strength_5": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low_loop_5": (lora_list, {"tooltip": "LoRA für LOW noise model Loop 5 (None = keine LoRA)"}),
                "lora_low_strength_5": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_loop_5": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 5 (harte Schnitt-Quelle)."}),
                "use_reference_loop_5": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 5"}),
                "use_overlap_loop_5": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 5"}),
                "use_endframe_loop_5": ("BOOLEAN", {"default": False, "tooltip": "Nutze Bild(er) des nächsten Loops als Endframe für Loop 5 (VACE Endframe)."}),
                "reference_image_5": ("IMAGE", {"tooltip": "Character-Referenz nur für Loop 5"}),

                # LOOP 6
                "prompt_loop_6": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 6 Prompt (leer = Base Prompt)"}),
                "lora_high_loop_6": (lora_list, {"tooltip": "LoRA für HIGH noise model Loop 6 (None = keine LoRA)"}),
                "lora_high_strength_6": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low_loop_6": (lora_list, {"tooltip": "LoRA für LOW noise model Loop 6 (None = keine LoRA)"}),
                "lora_low_strength_6": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_loop_6": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 6 (harte Schnitt-Quelle)."}),
                "use_reference_loop_6": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 6"}),
                "use_overlap_loop_6": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 6"}),
                "use_endframe_loop_6": ("BOOLEAN", {"default": False, "tooltip": "Nutze Bild(er) des nächsten Loops als Endframe für Loop 6 (VACE Endframe)."}),
                "reference_image_6": ("IMAGE", {"tooltip": "Character-Referenz nur für Loop 6"}),

                # LOOP 7
                "prompt_loop_7": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 7 Prompt (leer = Base Prompt)"}),
                "lora_high_loop_7": (lora_list, {"tooltip": "LoRA für HIGH noise model Loop 7 (None = keine LoRA)"}),
                "lora_high_strength_7": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low_loop_7": (lora_list, {"tooltip": "LoRA für LOW noise model Loop 7 (None = keine LoRA)"}),
                "lora_low_strength_7": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_loop_7": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 7 (harte Schnitt-Quelle)."}),
                "use_reference_loop_7": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 7"}),
                "use_overlap_loop_7": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 7"}),
                "use_endframe_loop_7": ("BOOLEAN", {"default": False, "tooltip": "Nutze Bild(er) des nächsten Loops als Endframe für Loop 7 (VACE Endframe)."}),
                "reference_image_7": ("IMAGE", {"tooltip": "Character-Referenz nur für Loop 7"}),

                # LOOP 8
                "prompt_loop_8": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 8 Prompt (leer = Base Prompt)"}),
                "lora_high_loop_8": (lora_list, {"tooltip": "LoRA für HIGH noise model Loop 8 (None = keine LoRA)"}),
                "lora_high_strength_8": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low_loop_8": (lora_list, {"tooltip": "LoRA für LOW noise model Loop 8 (None = keine LoRA)"}),
                "lora_low_strength_8": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_loop_8": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 8 (harte Schnitt-Quelle)."}),
                "use_reference_loop_8": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 8"}),
                "use_overlap_loop_8": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 8"}),
                "use_endframe_loop_8": ("BOOLEAN", {"default": False, "tooltip": "Nutze Bild(er) des nächsten Loops als Endframe für Loop 8 (VACE Endframe)."}),
                "reference_image_8": ("IMAGE", {"tooltip": "Character-Referenz nur für Loop 8"}),

                # LOOP 9
                "prompt_loop_9": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 9 Prompt (leer = Base Prompt)"}),
                "lora_high_loop_9": (lora_list, {"tooltip": "LoRA für HIGH noise model Loop 9 (None = keine LoRA)"}),
                "lora_high_strength_9": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low_loop_9": (lora_list, {"tooltip": "LoRA für LOW noise model Loop 9 (None = keine LoRA)"}),
                "lora_low_strength_9": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_loop_9": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 9 (harte Schnitt-Quelle)."}),
                "use_reference_loop_9": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 9"}),
                "use_overlap_loop_9": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 9"}),
                "use_endframe_loop_9": ("BOOLEAN", {"default": False, "tooltip": "Nutze Bild(er) des nächsten Loops als Endframe für Loop 9 (VACE Endframe)."}),
                "reference_image_9": ("IMAGE", {"tooltip": "Character-Referenz nur für Loop 9"}),

                # LOOP 10
                "prompt_loop_10": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 10 Prompt (leer = Base Prompt)"}),
                "lora_high_loop_10": (lora_list, {"tooltip": "LoRA für HIGH noise model Loop 10 (None = keine LoRA)"}),
                "lora_high_strength_10": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low_loop_10": (lora_list, {"tooltip": "LoRA für LOW noise model Loop 10 (None = keine LoRA)"}),
                "lora_low_strength_10": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_loop_10": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 10 (harte Schnitt-Quelle)."}),
                "use_reference_loop_10": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 10"}),
                "use_overlap_loop_10": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 10"}),
                "use_endframe_loop_10": ("BOOLEAN", {"default": False, "tooltip": "Nutze Bild(er) des nächsten Loops als Endframe für Loop 10 (falls vorhanden, sonst ignoriert)."}),
                "reference_image_10": ("IMAGE", {"tooltip": "Character-Referenz nur für Loop 10"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("full_video", "used_prompts")
    FUNCTION = "extend_video"
    CATEGORY = "WanCustom"

    def _safe_float(self, x, default=1.0):
        try:
            return float(x)
        except Exception:
            return default

    def _normalize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Normalize frame to [H, W, 3] in range 0-1"""
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame.squeeze(0)
        if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = frame.permute(1, 2, 0)

        if frame.min() < 0 or frame.max() > 1.5:
            frame = torch.clamp(frame, 0.0, 1.0)

        return frame

    def _tensor_to_frame_list(self, t: torch.Tensor):
        """
        Konvertiert einen IMAGE-Tensor in eine Liste von Frames [H, W, 3].
        """
        if t is None:
            return []

        if not isinstance(t, torch.Tensor):
            t = torch.from_numpy(np.array(t))

        t = t.detach().cpu().float()

        # (H, W, C)
        if t.ndim == 3:
            return [self._normalize_frame(t)]
        # (N, H, W, C)
        if t.ndim == 4 and t.shape[-1] in (1, 3):
            frames = []
            for i in range(t.shape[0]):
                frames.append(self._normalize_frame(t[i]))
            return frames
        # (C, H, W)
        if t.ndim == 3 and t.shape[0] in (1, 3):
            return [self._normalize_frame(t)]
        # (N, C, H, W)
        if t.ndim == 4 and t.shape[1] in (1, 3):
            frames = []
            for i in range(t.shape[0]):
                frames.append(self._normalize_frame(t[i]))
            return frames

        return [self._normalize_frame(t)]

    def _load_video_from_file(self, video_obj):
        """Extract frames from VideoFromFile object"""
        try:
            if hasattr(video_obj, "get_stream_source"):
                stream_source = video_obj.get_stream_source()

                if isinstance(stream_source, str):
                    video_path = stream_source

                    if os.path.exists(video_path):
                        cap = cv2.VideoCapture(video_path)
                        frames = []
                        try:
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame = frame.astype(np.float32) / 255.0
                                frames.append(torch.from_numpy(frame))
                        finally:
                            cap.release()

                        if frames:
                            return torch.stack(frames)
        except Exception as e:
            print(f"⚠ Error loading video: {e}")

        return None

    def _is_lora_none(self, lora_name):
        """Check if LoRA is set to None/disabled. 
        Handles None, empty strings, 'None' text, and missing values gracefully.
        """
        if lora_name is None:
            return True
        if not isinstance(lora_name, str):
            try:
                lora_name = str(lora_name)
            except Exception:
                return True
        lora_str = lora_name.strip().lower()
        return lora_str in ("", "none", "null", "disabled", "off")

    def _load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        """Load LoRA and patch model/clip. Returns original if None selected."""
        if self._is_lora_none(lora_name):
            return model, clip

        try:
            lora_name = str(lora_name)
            # Add .safetensors if missing
            if not lora_name.endswith(".safetensors"):
                lora_name = lora_name + ".safetensors"

            lora_path = folder_paths.get_full_path("loras", lora_name)

            if lora_path is None:
                print(f"⚠ LoRA not found: {lora_name}")
                return model, clip

            print(f"📦 Loading LoRA: {lora_name} (strength: {strength_model})")

            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora, strength_model, strength_clip
            )

            print("✓ LoRA loaded")
            return model_lora, clip_lora

        except Exception as e:
            print(f"⚠ Error loading LoRA {lora_name}: {e}")
            return model, clip

    def wan_vace_logic(self, vae, width, height, length, strength, control_video, control_masks, reference_image=None):
        """VACE-Logik mit optionalem Referenzbild (Char-Consistency).

        VRAM-optimiert:
        - control_video / mask bleiben auf CPU.
        - Nur die komprimierten Latents und die fertige Maske landen auf dem VAE-Device.
        """
        video_device = control_video.device

        base_latent_length = ((length - 1) // 4) + 1

        if control_masks is None:
            mask = torch.ones(
                (length, height, width, 1),
                device=video_device,
                dtype=control_video.dtype,
            )
        else:
            mask = control_masks.to(video_device)
            if mask.ndim == 3:
                mask = mask.unsqueeze(-1)

        # Pixel-Manipulationen vollständig auf CPU durchführen
        control_video_shifted = control_video - 0.5
        inactive_px = (control_video_shifted * (1.0 - mask)) + 0.5
        reactive_px = (control_video_shifted * mask) + 0.5

        # Encodierung mit VAE (VAE kümmert sich intern um Device)
        inactive_encoded = vae.encode(inactive_px[:, :, :, :3])
        reactive_encoded = vae.encode(reactive_px[:, :, :, :3])

        if isinstance(inactive_encoded, dict):
            inactive_encoded = inactive_encoded["samples"]
        if isinstance(reactive_encoded, dict):
            reactive_encoded = reactive_encoded["samples"]

        target_h = height // 8
        target_w = width // 8

        # Auf gewünschte Latentgröße bringen (T, H/8, W/8)
        if (
            inactive_encoded.shape[2] != base_latent_length
            or inactive_encoded.shape[3] != target_h
            or inactive_encoded.shape[4] != target_w
        ):
            inactive_encoded = torch.nn.functional.interpolate(
                inactive_encoded,
                size=(base_latent_length, target_h, target_w),
                mode="trilinear",
                align_corners=False,
            )
            reactive_encoded = torch.nn.functional.interpolate(
                reactive_encoded,
                size=(base_latent_length, target_h, target_w),
                mode="trilinear",
                align_corners=False,
            )

        control_video_latent = torch.cat([inactive_encoded, reactive_encoded], dim=1)
        print(f"Control video latent (before ref): {control_video_latent.shape}")

        # Optionales Referenzbild für Char-Consistency
        trim_latent = 0
        if reference_image is not None:
            print("🎨 Using reference image for character consistency")
            ref_resized = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)

            reference_encoded = vae.encode(ref_resized[:, :, :, :3])
            if isinstance(reference_encoded, dict):
                reference_encoded = reference_encoded["samples"]

            print(f"Reference encoded shape: {reference_encoded.shape}")

            reference_latent = torch.cat(
                [
                    reference_encoded,
                    comfy.latent_formats.Wan21().process_out(
                        torch.zeros_like(reference_encoded)
                    ),
                ],
                dim=1,
            )

            print(f"Reference latent shape: {reference_latent.shape}")

            control_video_latent = torch.cat(
                [reference_latent, control_video_latent], dim=2
            )
            print(f"Control video latent (after ref): {control_video_latent.shape}")

            trim_latent = reference_latent.shape[2]

        latent_length = base_latent_length + trim_latent
        print(
            f"Final latent_length: {latent_length} (base: {base_latent_length}, trim: {trim_latent})"
        )

        # Maske in VAE-Auflösung bringen
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride

        mask_proc = mask.squeeze(-1)
        mask_proc = mask_proc.view(
            length, height_mask, vae_stride, width_mask, vae_stride
        )
        mask_proc = mask_proc.permute(2, 4, 0, 1, 3)
        mask_proc = mask_proc.reshape(
            vae_stride * vae_stride, length, height_mask, width_mask
        )

        mask_proc = torch.nn.functional.interpolate(
            mask_proc.unsqueeze(0),
            size=(base_latent_length, height_mask, width_mask),
            mode="nearest-exact",
        ).squeeze(0)

        print(f"Mask after interpolation (base): {mask_proc.shape}")

        if trim_latent > 0:
            mask_pad = torch.zeros(
                (mask_proc.shape[0], trim_latent, height_mask, width_mask),
                device=mask_proc.device,
                dtype=mask_proc.dtype,
            )
            mask_proc = torch.cat([mask_pad, mask_proc], dim=1)
            print(f"Mask with reference padding: {mask_proc.shape}")

        mask_proc = mask_proc.unsqueeze(0)
        mask_proc = mask_proc.to(control_video_latent.device)

        print(f"Final mask shape: {mask_proc.shape}")

        return control_video_latent, mask_proc, trim_latent

    def extend_video(
        self,
        high_noise,
        low_noise,
        vae,
        clip,
        positive_prompt,
        negative_prompt,
        sampler_name_high,
        scheduler_high,
        steps_h,
        start_at_step_h,
        end_at_step_h,
        cfg_h,
        sampler_name_low,
        scheduler_low,
        steps_l,
        start_at_step_l,
        end_at_step_l,
        cfg_l,
        extension_loops,
        generate_frames,
        overlap_frames,
        empty_frame_level,
        strength,
        seed,
        default_width,
        default_height,
        image=None,
        video=None,
        reference_image=None,
        inpaint_mask=None,
        # Loop Prompts
        prompt_loop_1="",
        prompt_loop_2="",
        prompt_loop_3="",
        prompt_loop_4="",
        prompt_loop_5="",
        prompt_loop_6="",
        prompt_loop_7="",
        prompt_loop_8="",
        prompt_loop_9="",
        prompt_loop_10="",
        # Loop LoRAs HIGH (Dropdown)
        lora_high_loop_1="None",
        lora_high_strength_1=1.0,
        lora_high_loop_2="None",
        lora_high_strength_2=1.0,
        lora_high_loop_3="None",
        lora_high_strength_3=1.0,
        lora_high_loop_4="None",
        lora_high_strength_4=1.0,
        lora_high_loop_5="None",
        lora_high_strength_5=1.0,
        lora_high_loop_6="None",
        lora_high_strength_6=1.0,
        lora_high_loop_7="None",
        lora_high_strength_7=1.0,
        lora_high_loop_8="None",
        lora_high_strength_8=1.0,
        lora_high_loop_9="None",
        lora_high_strength_9=1.0,
        lora_high_loop_10="None",
        lora_high_strength_10=1.0,
        # Loop LoRAs LOW (Dropdown)
        lora_low_loop_1="None",
        lora_low_strength_1=1.0,
        lora_low_loop_2="None",
        lora_low_strength_2=1.0,
        lora_low_loop_3="None",
        lora_low_strength_3=1.0,
        lora_low_loop_4="None",
        lora_low_strength_4=1.0,
        lora_low_loop_5="None",
        lora_low_strength_5=1.0,
        lora_low_loop_6="None",
        lora_low_strength_6=1.0,
        lora_low_loop_7="None",
        lora_low_strength_7=1.0,
        lora_low_loop_8="None",
        lora_low_strength_8=1.0,
        lora_low_loop_9="None",
        lora_low_strength_9=1.0,
        lora_low_loop_10="None",
        lora_low_strength_10=1.0,
        # Loop Images
        image_loop_1=None,
        image_loop_2=None,
        image_loop_3=None,
        image_loop_4=None,
        image_loop_5=None,
        image_loop_6=None,
        image_loop_7=None,
        image_loop_8=None,
        image_loop_9=None,
        image_loop_10=None,
        # Loop Reference Toggles
        use_reference_loop_1=False,
        use_reference_loop_2=False,
        use_reference_loop_3=False,
        use_reference_loop_4=False,
        use_reference_loop_5=False,
        use_reference_loop_6=False,
        use_reference_loop_7=False,
        use_reference_loop_8=False,
        use_reference_loop_9=False,
        use_reference_loop_10=False,
        # Loop Overlap Toggles
        use_overlap_loop_1=False,
        use_overlap_loop_2=False,
        use_overlap_loop_3=False,
        use_overlap_loop_4=False,
        use_overlap_loop_5=False,
        use_overlap_loop_6=False,
        use_overlap_loop_7=False,
        use_overlap_loop_8=False,
        use_overlap_loop_9=False,
        use_overlap_loop_10=False,
        # Loop Endframe Toggles
        use_endframe_loop_1=False,
        use_endframe_loop_2=False,
        use_endframe_loop_3=False,
        use_endframe_loop_4=False,
        use_endframe_loop_5=False,
        use_endframe_loop_6=False,
        use_endframe_loop_7=False,
        use_endframe_loop_8=False,
        use_endframe_loop_9=False,
        use_endframe_loop_10=False,
        # Loop reference_image
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        reference_image_5=None,
        reference_image_6=None,
        reference_image_7=None,
        reference_image_8=None,
        reference_image_9=None,
        reference_image_10=None,
    ):

        print("\n" + "=" * 60)
        print("WAN VIDEO EXTENDER PRO high+low-noise (Memory Optimized + Endframe)")
        print("=" * 60)

        loop_prompts = [
            prompt_loop_1,
            prompt_loop_2,
            prompt_loop_3,
            prompt_loop_4,
            prompt_loop_5,
            prompt_loop_6,
            prompt_loop_7,
            prompt_loop_8,
            prompt_loop_9,
            prompt_loop_10,
        ]

        # (high_name, high_strength, low_name, low_strength)
        loop_loras = [
            (lora_high_loop_1, lora_high_strength_1, lora_low_loop_1, lora_low_strength_1),
            (lora_high_loop_2, lora_high_strength_2, lora_low_loop_2, lora_low_strength_2),
            (lora_high_loop_3, lora_high_strength_3, lora_low_loop_3, lora_low_strength_3),
            (lora_high_loop_4, lora_high_strength_4, lora_low_loop_4, lora_low_strength_4),
            (lora_high_loop_5, lora_high_strength_5, lora_low_loop_5, lora_low_strength_5),
            (lora_high_loop_6, lora_high_strength_6, lora_low_loop_6, lora_low_strength_6),
            (lora_high_loop_7, lora_high_strength_7, lora_low_loop_7, lora_low_strength_7),
            (lora_high_loop_8, lora_high_strength_8, lora_low_loop_8, lora_low_strength_8),
            (lora_high_loop_9, lora_high_strength_9, lora_low_loop_9, lora_low_strength_9),
            (lora_high_loop_10, lora_high_strength_10, lora_low_loop_10, lora_low_strength_10),
        ]

        loop_images = [
            image_loop_1,
            image_loop_2,
            image_loop_3,
            image_loop_4,
            image_loop_5,
            image_loop_6,
            image_loop_7,
            image_loop_8,
            image_loop_9,
            image_loop_10,
        ]

        loop_use_reference = [
            use_reference_loop_1, use_reference_loop_2, use_reference_loop_3,
            use_reference_loop_4, use_reference_loop_5, use_reference_loop_6,
            use_reference_loop_7, use_reference_loop_8, use_reference_loop_9,
            use_reference_loop_10,
        ]

        loop_use_overlap = [
            use_overlap_loop_1, use_overlap_loop_2, use_overlap_loop_3,
            use_overlap_loop_4, use_overlap_loop_5, use_overlap_loop_6,
            use_overlap_loop_7, use_overlap_loop_8, use_overlap_loop_9,
            use_overlap_loop_10,
        ]

        loop_use_endframe = [
            use_endframe_loop_1, use_endframe_loop_2, use_endframe_loop_3,
            use_endframe_loop_4, use_endframe_loop_5, use_endframe_loop_6,
            use_endframe_loop_7, use_endframe_loop_8, use_endframe_loop_9,
            use_endframe_loop_10,
        ]

        loop_reference_images = [
            reference_image_1,
            reference_image_2,
            reference_image_3,
            reference_image_4,
            reference_image_5,
            reference_image_6,
            reference_image_7,
            reference_image_8,
            reference_image_9,
            reference_image_10,
        ]

        used_prompts_log = []

        segment_dir = tempfile.mkdtemp(prefix="wan_segments_")
        print(f"🗂 Using temp segment dir: {segment_dir}")

        segment_paths = []
        original_input_segment_path = None

        # ---------------------------
        # Eingangsframes bestimmen
        # ---------------------------
        initial_frames = None

        # 1) Globales Bild / Video
        if image is not None:
            print("Input: IMAGE")
            initial_frames = image
        elif video is not None:
            print("Input: VIDEO")
            if isinstance(video, torch.Tensor):
                initial_frames = video
            else:
                initial_frames = self._load_video_from_file(video)

        # 2) Fallback: Erstes verfügbares Loop-Image als Eingangsframes
        if initial_frames is None:
            for li in loop_images:
                if li is not None:
                    loop_frames = self._tensor_to_frame_list(li)
                    if len(loop_frames) > 0:
                        initial_frames = torch.stack(loop_frames)
                        print(f"Input: using first loop_image as initial input ({initial_frames.shape[0]} frames)")
                        break

        has_initial_input_from_user = initial_frames is not None

        context_frames = []
        H, W = default_height, default_width

        if initial_frames is not None:
            if isinstance(initial_frames, torch.Tensor) and initial_frames.ndim == 3:
                initial_frames = initial_frames.unsqueeze(0)
            if isinstance(initial_frames, torch.Tensor):
                initial_frames = initial_frames.cpu().float()
            else:
                initial_frames = torch.from_numpy(np.array(initial_frames)).float()

            num_input_frames = initial_frames.shape[0]
            print(f"✓ Loaded {num_input_frames} input frames")

            original_input_segment_path = os.path.join(segment_dir, "segment_input.pt")
            try:
                torch.save(initial_frames, original_input_segment_path)
                print(f"💾 Saved original input segment: {original_input_segment_path}")
            except Exception as e:
                print(f"⚠ Failed to save original input segment: {e}")
                original_input_segment_path = None

            slice_len = min(num_input_frames, overlap_frames)
            if slice_len > 0:
                last_slice = initial_frames[-slice_len:]
                for i in range(slice_len):
                    frame = self._normalize_frame(last_slice[i])
                    context_frames.append(frame.cpu())

            if initial_frames.ndim == 4 and initial_frames.shape[-1] == 3:
                H, W = initial_frames.shape[1], initial_frames.shape[2]
            elif initial_frames.ndim == 4 and initial_frames.shape[1] == 3:
                H, W = initial_frames.shape[2], initial_frames.shape[3]

            del initial_frames
            gc.collect()

        print(f"Resolution: {W}x{H}")
        print(f"Initial context frames: {len(context_frames)}")

        current_seed = seed

        # Store base models (for clean LoRA patching each loop)
        base_high_model = high_noise
        base_low_model = low_noise
        base_clip = clip

        # === MAIN LOOP ===
        for loop_idx in range(extension_loops):
            loop_id = loop_idx + 1

            # Cleanup before each loop
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print("\n" + "─" * 60)
            print(f"LOOP {loop_id}/{extension_loops}")
            if torch.cuda.is_available():
                print(
                    f"VRAM: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB (vor Loop)"
                )
            print("─" * 60)

            # LoRA pro Loop: separate LORAs für HIGH und LOW model
            try:
                if loop_idx < len(loop_loras):
                    lora_high_name, lora_high_str, lora_low_name, lora_low_str = loop_loras[loop_idx]
                    
                    # HIGH noise model LoRA
                    if not self._is_lora_none(lora_high_name):
                        lora_high_str = self._safe_float(lora_high_str, 1.0)
                        high_noise, clip_high = self._load_lora(
                            base_high_model, base_clip, lora_high_name, lora_high_str, lora_high_str
                        )
                        print(f"🎨 Loop {loop_id}: HIGH LoRA '{lora_high_name}' loaded (strength: {lora_high_str})")
                    else:
                        high_noise = base_high_model
                        clip_high = base_clip
                        print(f"🎨 Loop {loop_id}: No HIGH LoRA (None selected)")

                    # LOW noise model LoRA
                    if not self._is_lora_none(lora_low_name):
                        lora_low_str = self._safe_float(lora_low_str, 1.0)
                        low_noise, clip_low = self._load_lora(
                            base_low_model, clip_high, lora_low_name, lora_low_str, lora_low_str
                        )
                        clip = clip_low  # Use clip from low LoRA patching
                        print(f"🎨 Loop {loop_id}: LOW LoRA '{lora_low_name}' loaded (strength: {lora_low_str})")
                    else:
                        low_noise = base_low_model
                        clip = clip_high  # Use clip from high LoRA patching (or base if no high lora)
                        print(f"🎨 Loop {loop_id}: No LOW LoRA (None selected)")
                else:
                    high_noise = base_high_model
                    low_noise = base_low_model
                    clip = base_clip
            except Exception as e:
                # LoRA loading failed - continue with base models
                high_noise = base_high_model
                low_noise = base_low_model
                clip = base_clip
                print(f"⚠️ Loop {loop_id}: LoRA loading failed ({e}), using base models")

            loop_prompt_raw = ""
            if loop_idx < len(loop_prompts):
                loop_prompt_raw = (loop_prompts[loop_idx] or "").strip()

            if loop_prompt_raw:
                active_prompt = loop_prompt_raw
                print(f"📝 Using Loop {loop_id} Prompt (input)")
            else:
                active_prompt = positive_prompt
                print("📝 Using Base Prompt")

            used_prompts_log.append(f"Loop {loop_id}: {active_prompt[:80]}")

            def get_cond(text, clip_obj):
                tokens = clip_obj.tokenize(text)
                cond, pooled = clip_obj.encode_from_tokens(tokens, return_pooled=True)
                return [[cond, {"pooled_output": pooled}]]

            c_pos = get_cond(active_prompt, clip)
            c_neg = get_cond(negative_prompt, clip)

            loop_image = loop_images[loop_idx] if loop_idx < len(loop_images) else None
            loop_image_frames = []

            if loop_image is not None:
                loop_image_frames = self._tensor_to_frame_list(loop_image)
                if len(loop_image_frames) > 0:
                    print(
                        f"🎬 Loop {loop_id}: using {len(loop_image_frames)} loop image frame(s) as CUT context."
                    )
                    base_context_candidate = loop_image_frames
                else:
                    base_context_candidate = context_frames
            else:
                base_context_candidate = context_frames

            if len(base_context_candidate) > 0:
                max_context = min(len(base_context_candidate), overlap_frames, generate_frames)
                selected_context_frames = base_context_candidate[-max_context:]
            else:
                selected_context_frames = []

            if len(selected_context_frames) > 0:
                H, W, C = selected_context_frames[0].shape
            else:
                H, W = default_height, default_width

            print(
                f"Frame size: {H}x{W}, Context frames this loop: {len(selected_context_frames)}"
            )

            full_pixels = torch.ones(
                (generate_frames, H, W, 3), device="cpu"
            ) * empty_frame_level
            full_masks = torch.ones((generate_frames, H, W), device="cpu")

            context_count = 0
            if len(selected_context_frames) > 0:
                context_batch = torch.stack(selected_context_frames)
                context_count = context_batch.shape[0]

                has_loop_image = loop_image is not None and len(loop_image_frames) > 0

                if loop_idx == 0:
                    # Loop 1: immer Kontext nutzen (Input oder image_loop_1)
                    write_context = True
                    if has_loop_image:
                        print(f"🚀 Loop 1: Using image_loop_1 as hard-cut context ({context_count} frames)")
                    else:
                        print(f"🚀 Loop 1: Using initial input as context ({context_count} frames)")
                else:
                    if has_loop_image:
                        # Harte Schnitte: image_loop_X immer verwenden, unabhängig vom Overlap-Switch
                        write_context = True
                        print(f"✂️ Loop {loop_id}: Using loop image(s) as hard-cut context ({context_count} frames)")
                    else:
                        # Kein loop_image -> nur Overlap aus vorherigen Loops, gesteuert vom User
                        write_context = bool(loop_use_overlap[loop_idx])
                        if write_context:
                            print(f"🔁 Loop {loop_id}: Using overlap context ({context_count} frames)")
                        else:
                            print(f"⛔ Loop {loop_id}: Context/Overlap disabled (no loop_image)")

                if write_context:
                    full_pixels[:context_count] = context_batch
                    full_masks[:context_count] = 0.0  # fixer Kontext
                    print(f"✅ Context written to canvas: {context_count} frames")
                else:
                    context_count = 0
            else:
                context_batch = None

            # === ENDFRAME LOGIC (VACE Start-to-End) ===
            endframe_count = 0
            if loop_use_endframe[loop_idx]:
                # Get image from NEXT loop as endframe target
                next_loop_idx = loop_idx + 1
                if next_loop_idx < len(loop_images) and loop_images[next_loop_idx] is not None:
                    endframe_source = loop_images[next_loop_idx]
                    endframe_frames = self._tensor_to_frame_list(endframe_source)
                    
                    if len(endframe_frames) > 0:
                        # Resize endframes to match current resolution if needed
                        resized_endframes = []
                        for ef in endframe_frames:
                            if ef.shape[0] != H or ef.shape[1] != W:
                                # Resize using torch interpolate
                                ef_resized = torch.nn.functional.interpolate(
                                    ef.permute(2, 0, 1).unsqueeze(0),
                                    size=(H, W),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0).permute(1, 2, 0)
                                resized_endframes.append(ef_resized)
                            else:
                                resized_endframes.append(ef)
                        
                        endframe_batch = torch.stack(resized_endframes)
                        endframe_count = min(endframe_batch.shape[0], generate_frames - context_count - 1)
                        
                        if endframe_count > 0:
                            # Place endframes at the END of full_pixels
                            end_start_idx = generate_frames - endframe_count
                            full_pixels[end_start_idx:] = endframe_batch[:endframe_count]
                            full_masks[end_start_idx:] = 0.0
                            print(f"🎯 Loop {loop_id}: Endframe enabled - using {endframe_count} frame(s) from Loop {next_loop_idx + 1} as target")
                        else:
                            print(f"⚠️ Loop {loop_id}: Endframe requested but no space left after context")
                    else:
                        print(f"⚠️ Loop {loop_id}: Endframe enabled but next loop has no valid image")
                else:
                    print(f"⚠️ Loop {loop_id}: Endframe enabled but no image connected to Loop {next_loop_idx + 1}")

            # Referenz pro Loop
            current_loop_reference = None
            if loop_reference_images[loop_idx] is not None:
                current_loop_reference = loop_reference_images[loop_idx]
                print(f"🎨 Loop {loop_id}: Using local loop reference image.")
            elif reference_image is not None and loop_use_reference[loop_idx]:
                current_loop_reference = reference_image
                print(f"🎨 Loop {loop_id}: Using global reference image (switch enabled).")
            else:
                print(f"🎨 Loop {loop_id}: No reference image used.")

            vace_latents, vace_masks, trim_latent = self.wan_vace_logic(
                vae=vae,
                width=W,
                height=H,
                length=generate_frames,
                strength=strength,
                control_video=full_pixels,
                control_masks=full_masks,
                reference_image=current_loop_reference,
            )

            print(
                f"VACE returned - latents: {vace_latents.shape}, masks: {vace_masks.shape}, trim: {trim_latent}"
            )

            new_pos = node_helpers.conditioning_set_values(
                c_pos,
                {
                    "vace_frames": [vace_latents],
                    "vace_mask": [vace_masks],
                    "vace_strength": [strength],
                },
                append=True,
            )

            new_neg = node_helpers.conditioning_set_values(
                c_neg,
                {
                    "vace_frames": [vace_latents],
                    "vace_mask": [vace_masks],
                    "vace_strength": [strength],
                },
                append=True,
            )

            latent_length = ((generate_frames - 1) // 4) + 1 + trim_latent
            empty_latent = torch.zeros(
                [1, 16, latent_length, H // 8, W // 8],
                device=comfy.model_management.get_torch_device(),
            )
            latents = {"samples": empty_latent}

            print(f"Empty latent shape: {empty_latent.shape}")

            print("🎬 Sampling... (HIGH noise)")
            ksampler_adv = nodes.KSamplerAdvanced()

            high_out = ksampler_adv.sample(
                model=high_noise,
                add_noise="enable",
                noise_seed=current_seed,
                steps=steps_h,
                cfg=cfg_h,
                sampler_name=sampler_name_high,
                scheduler=scheduler_high,
                positive=new_pos,
                negative=new_neg,
                latent_image=latents,
                start_at_step=start_at_step_h,
                end_at_step=end_at_step_h,
                return_with_leftover_noise="enable",
            )
            high_sample = {"samples": high_out[0]["samples"]}

            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print("🎬 Sampling... (LOW noise)")
            low_sample = ksampler_adv.sample(
                model=low_noise,
                add_noise="disable",
                noise_seed=current_seed,
                steps=steps_l,
                cfg=cfg_l,
                sampler_name=sampler_name_low,
                scheduler=scheduler_low,
                positive=new_pos,
                negative=new_neg,
                latent_image=high_sample,
                start_at_step=start_at_step_l,
                end_at_step=end_at_step_l,
                return_with_leftover_noise="disable",
            )
            new_samples = low_sample[0]["samples"]

            # Trim reference frames if present
            if trim_latent > 0:
                new_samples = new_samples[:, :, trim_latent:, :, :]

            decoded = vae.decode(new_samples)
            decoded = decoded.cpu()

            if decoded.dim() == 5:
                decoded = decoded[0]

            print(f"✓ Decoded: {decoded.shape}")

            del new_samples, vace_latents, vace_masks, empty_latent, latents
            del c_pos, c_neg, new_pos, new_neg, context_batch
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # === ADD FRAMES AS DISK SEGMENT (ONLY NEW PART, Kontext + Endframes überspringen) ===
            skip_count = context_count
            end_trim = endframe_count  # Trim endframes from output
            new_frames_list = []
            frame_end_idx = decoded.shape[0] - end_trim
            for j in range(skip_count, frame_end_idx):
                frame = self._normalize_frame(decoded[j]).cpu()
                new_frames_list.append(frame)

            new_frames = len(new_frames_list)
            if end_trim > 0:
                print(f"✓ New frames this loop: {new_frames} (skipped {skip_count} start, {end_trim} end)")
            else:
                print(f"✓ New frames this loop (without overlap/context): {new_frames}")

            if new_frames > 0:
                segment_tensor = torch.stack(new_frames_list)
                segment_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
                try:
                    torch.save(segment_tensor, segment_path)
                    segment_paths.append(segment_path)
                    print(f"💾 Saved segment {loop_id} -> {segment_path}")
                except Exception as e:
                    print(f"⚠ Failed to save segment {loop_id}: {e}")

                # Update context for next loop:
                # letzte overlap_frames aus [Kontext dieses Loops + neuen Frames]
                combined_tail = selected_context_frames + new_frames_list
                if len(combined_tail) > overlap_frames:
                    context_frames = combined_tail[-overlap_frames:]
                else:
                    context_frames = combined_tail
            else:
                print("⚠ No new frames generated in this loop, keeping previous context_frames")

            del decoded, new_frames_list
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            current_seed += 1

        # === FINAL COMBINE FROM DISK SEGMENTS (MEMORY OPTIMIZED) ===
        print("\n" + "=" * 60)
        print("COMPLETE - combining segments from disk")
        print("=" * 60 + "\n")

        full_video = None

        # 1) If user provided input (image or video), load original input segment first
        if has_initial_input_from_user and original_input_segment_path is not None:
            try:
                full_video = torch.load(original_input_segment_path, map_location="cpu")
                print(f"📥 Loaded original input segment ({full_video.shape[0]} frames)")
            except Exception as e:
                print(f"⚠ Failed to load original input segment: {e}")

        # 2) Append all generated segments incrementally (memory efficient!)
        total_generated = 0
        for idx, p in enumerate(segment_paths):
            try:
                print(f"📥 Loading segment {os.path.basename(p)}...")
                seg = torch.load(p, map_location="cpu")
                seg_frames = seg.shape[0]

                if full_video is None:
                    full_video = seg
                else:
                    full_video = torch.cat([full_video, seg], dim=0)
                    del seg  # ⭐ Immediate cleanup!
                    gc.collect()

                total_generated += seg_frames
                print(f"  ✓ Total frames now: {full_video.shape[0]}")
            except Exception as e:
                print(f"⚠ Failed to load segment {p}: {e}")

        # 3) If no segments at all, create empty fallback
        if full_video is None:
            full_video = torch.zeros((1, default_height, default_width, 3))

        # Cleanup temp directory
        try:
            shutil.rmtree(segment_dir)
            print(f"🧹 Cleaned temp dir: {segment_dir}")
        except Exception as e:
            print(f"⚠ Could not remove temp dir {segment_dir}: {e}")

        return (full_video, "\n".join(used_prompts_log))


NODE_CLASS_MAPPINGS = {
    "WanVideoExtenderLowHigh": WanVideoExtenderLowHigh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoExtenderLowHigh": "Wan 2.2 Video Extender PRO high+low-noise"
}