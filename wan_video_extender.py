import torch
import numpy as np
import comfy.model_management
import comfy.sample
import comfy.utils
import nodes
import node_helpers
import folder_paths
import cv2
import os
import copy
import tempfile
import shutil
import gc


class WanVideoExtenderNative:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),

                # MAIN PROMPTS
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": " ",
                    "placeholder": "Main prompt for a single run, image-to-video, or video extension input. If using more than two connections, select the loop input for the string and leave this field empty. Keep it short, this is only a basic template.",
                    "tooltip": "Main-Prompt / Basis-Prompt for all Loops (Fallback)."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "low quality, blurry, glitch, distortion"
                }),

                # SETTINGS
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),

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
                "reference_image": ("IMAGE", {"tooltip": "Character Reference für VACE"}),
                "inpaint_mask": ("MASK",),

                # LOOP 1: Prompt + LoRA + Image
                "prompt_loop_1": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Optionaler Loop-1-Prompt (String-Node anschließen, leer = Base Prompt)."
                }),
                "lora_loop_1": ("STRING", {
                    "default": "",
                    "tooltip": "Loop 1 LoRA Name (ohne .safetensors, optional über String-Node)."
                }),
                "lora_strength_1": ("STRING", {"default": "1.0", "multiline": False}),
                "image_loop_1": ("IMAGE", {
                    "tooltip": "Optionales Bild / Bilder für Loop 1. Wenn gesetzt, wird hier ein harter Schnitt gemacht und diese Frames als Kontext verwendet (statt vorheriger 16 Frames)."
                }),
                "reference_image_1": ("IMAGE", {"tooltip": "Character Reference für VACE"}),

                # LOOP 2
                "prompt_loop_2": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 2 Prompt (leer = Base Prompt)"}),
                "lora_loop_2": ("STRING", {"default": ""}),
                "lora_strength_2": ("STRING", {"default": "1.0", "multiline": False}),
                "image_loop_2": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 2 (harte Schnitt-Quelle)."}),
                "use_reference_loop_1": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 1"}),
                "use_overlap_loop_1": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 1"}),
                "reference_image_2": ("IMAGE", {"tooltip": "Character Reference für VACE"}),

                # LOOP 3
                "prompt_loop_3": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 3 Prompt (leer = Base Prompt)"}),
                "lora_loop_3": ("STRING", {"default": ""}),
                "lora_strength_3": ("STRING", {"default": "1.0", "multiline": False}),
                "image_loop_3": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 3 (harte Schnitt-Quelle)."}),
                "use_reference_loop_2": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 2"}),
                "use_overlap_loop_2": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 2"}),
                "reference_image_3": ("IMAGE", {"tooltip": "Character Reference für VACE"}),
                

                # LOOP 4
                "prompt_loop_4": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 4 Prompt (leer = Base Prompt)"}),
                "lora_loop_4": ("STRING", {"default": ""}),
                "lora_strength_4": ("STRING", {"default": "1.0", "multiline": False}),
                "image_loop_4": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 4 (harte Schnitt-Quelle)."}),
                "use_reference_loop_3": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 3"}),
                "use_overlap_loop_3": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 3"}),
                "reference_image_4": ("IMAGE", {"tooltip": "Character Reference für VACE"}),

                # LOOP 5
                "prompt_loop_5": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 5 Prompt (leer = Base Prompt)"}),
                "lora_loop_5": ("STRING", {"default": ""}),
                "lora_strength_5": ("STRING", {"default": "1.0", "multiline": False}),
                "image_loop_5": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 5 (harte Schnitt-Quelle)."}),
                "use_reference_loop_4": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 4"}),
                "use_overlap_loop_4": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 4"}),
                "reference_image_5": ("IMAGE", {"tooltip": "Character Reference für VACE"}),

                # LOOP 6
                "prompt_loop_6": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 6 Prompt (leer = Base Prompt)"}),
                "lora_loop_6": ("STRING", {"default": ""}),
                "lora_strength_6": ("STRING", {"default": "1.0", "multiline": False}),
                "image_loop_6": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 6 (harte Schnitt-Quelle)."}),
                "use_reference_loop_5": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 5"}),
                "use_overlap_loop_5": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 5"}),
                "reference_image_6": ("IMAGE", {"tooltip": "Character Reference für VACE"}),

                # LOOP 7
                "prompt_loop_7": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 7 Prompt (leer = Base Prompt)"}),
                "lora_loop_7": ("STRING", {"default": ""}),
                "lora_strength_7": ("STRING", {"default": "1.0", "multiline": False}),
                "image_loop_7": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 7 (harte Schnitt-Quelle)."}),
                "use_reference_loop_6": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 6"}),
                "use_overlap_loop_6": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 6"}),
                "reference_image_7": ("IMAGE", {"tooltip": "Character Reference für VACE"}),

                # LOOP 8
                "prompt_loop_8": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 8 Prompt (leer = Base Prompt)"}),
                "lora_loop_8": ("STRING", {"default": ""}),
                "lora_strength_8": ("STRING", {"default": "1.0", "multiline": False}),
                "image_loop_8": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 8 (harte Schnitt-Quelle)."}),
                "use_reference_loop_7": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 7"}),
                "use_overlap_loop_7": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 7"}),
                "reference_image_8": ("IMAGE", {"tooltip": "Character Reference für VACE"}),

                # LOOP 9
                "prompt_loop_9": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 9 Prompt (leer = Base Prompt)"}),
                "lora_loop_9": ("STRING", {"default": ""}),
                "lora_strength_9": ("STRING", {"default": "1.0", "multiline": False}),
                "image_loop_9": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 9 (harte Schnitt-Quelle)."}),
                "use_reference_loop_8": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 8"}),
                "use_overlap_loop_8": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 8"}),
                "reference_image_9": ("IMAGE", {"tooltip": "Character Reference für VACE"}),

                # LOOP 10
                "prompt_loop_10": ("STRING", {"multiline": False, "default": "", "tooltip": "Loop 10 Prompt (leer = Base Prompt)"}),
                "lora_loop_10": ("STRING", {"default": ""}),
                "lora_strength_10": ("STRING", {"default": "1.0", "multiline": False}),
                "image_loop_10": ("IMAGE", {"tooltip": "Optionales Bild / Bilder für Loop 10 (harte Schnitt-Quelle)."}),
                "use_reference_loop_9": ("BOOLEAN", {"default": False, "tooltip": "Use global reference image for Loop 9"}),
                "use_overlap_loop_9": ("BOOLEAN", {"default": False, "tooltip": "Use overlap context for Loop 9"}),
                "reference_image_10": ("IMAGE", {"tooltip": "Character Reference für VACE"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("full_video", "used_prompts")
    FUNCTION = "extend_video"
    CATEGORY = "WanCustom"
    
    def _safe_float(self, x, default=1.0):
        try:
            return float(x)
        except:
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
        Nimmt ein IMAGE-Tensor in beliebigem der üblichen Formate und gibt eine Liste von Frames [H, W, 3] zurück.
        """
        if t is None:
            return []

        if not isinstance(t, torch.Tensor):
            t = torch.from_numpy(np.array(t))

        # bring to float32 on CPU
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

        # Fallback: versuchen als (H, W, C)
        return [self._normalize_frame(t)]

    def _load_video_from_file(self, video_obj):
        """Extract frames from VideoFromFile object"""
        try:
            if hasattr(video_obj, 'get_stream_source'):
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

    def _load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        """Load LoRA and patch model/clip"""
        if not lora_name or str(lora_name).strip() == "":
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

            # Patch model
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora, strength_model, strength_clip
            )

            print("✓ LoRA loaded")
            return model_lora, clip_lora

        except Exception as e:
            print(f"⚠ Error loading LoRA {lora_name}: {e}")
            return model, clip

    def wan_vace_logic(self, vae, width, height, length, strength, control_video, control_masks, reference_image=None):
        """VACE Logic with optional Reference Image (unverändert)."""
        device = comfy.model_management.get_torch_device()

        # Calculate base latent length (without reference)
        base_latent_length = ((length - 1) // 4) + 1
        control_video = control_video.to(device)

        if control_masks is None:
            mask = torch.ones((length, height, width, 1), device=device)
        else:
            mask = control_masks.to(device)
            if mask.ndim == 3:
                mask = mask.unsqueeze(-1)

        # Inactive / Reactive Split
        control_video_shifted = control_video - 0.5
        inactive_px = (control_video_shifted * (1.0 - mask)) + 0.5
        reactive_px = (control_video_shifted * mask) + 0.5

        # VAE Encoding
        inactive_encoded = vae.encode(inactive_px[:, :, :, :3])
        reactive_encoded = vae.encode(reactive_px[:, :, :, :3])

        if isinstance(inactive_encoded, dict):
            inactive_encoded = inactive_encoded["samples"]
        if isinstance(reactive_encoded, dict):
            reactive_encoded = reactive_encoded["samples"]

        # Interpolate to base latent size (without reference)
        target_h = height // 8
        target_w = width // 8

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

        # Concatenate inactive + reactive first
        control_video_latent = torch.cat([inactive_encoded, reactive_encoded], dim=1)
        print(f"Control video latent (before ref): {control_video_latent.shape}")

        # Handle Reference Image
        trim_latent = 0
        if reference_image is not None:
            print("🎨 Using reference image for character consistency")
            # Resize reference to target size
            ref_resized = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)

            # Encode reference
            reference_encoded = vae.encode(ref_resized[:, :, :, :3])
            if isinstance(reference_encoded, dict):
                reference_encoded = reference_encoded["samples"]

            print(f"Reference encoded shape: {reference_encoded.shape}")

            # Process reference (add zeros channel to match WAN format)
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

            # Concatenate in TIME dimension (dim=2)
            control_video_latent = torch.cat(
                [reference_latent, control_video_latent], dim=2
            )
            print(f"Control video latent (after ref): {control_video_latent.shape}")

            # Store how many frames the reference takes
            trim_latent = reference_latent.shape[2]

        # NOW calculate final latent_length
        latent_length = base_latent_length + trim_latent
        print(
            f"Final latent_length: {latent_length} (base: {base_latent_length}, trim: {trim_latent})"
        )

        # Mask Permutation - use BASE latent_length (without reference)
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

        # Interpolate to BASE latent length (without reference frames yet!)
        mask_proc = torch.nn.functional.interpolate(
            mask_proc.unsqueeze(0),
            size=(base_latent_length, height_mask, width_mask),
            mode="nearest-exact",
        ).squeeze(0)

        print(f"Mask after interpolation (base): {mask_proc.shape}")

        # Add padding for reference frames at the BEGINNING
        if trim_latent > 0:
            mask_pad = torch.zeros(
                (mask_proc.shape[0], trim_latent, height_mask, width_mask),
                device=mask_proc.device,
                dtype=mask_proc.dtype,
            )
            mask_proc = torch.cat([mask_pad, mask_proc], dim=1)
            print(f"Mask with reference padding: {mask_proc.shape}")

        mask_proc = mask_proc.unsqueeze(0)
        print(f"Final mask shape: {mask_proc.shape}")

        return control_video_latent, mask_proc, trim_latent

    def extend_video(
        self,
        model,
        vae,
        clip,
        positive_prompt,
        negative_prompt,
        steps,
        cfg,
        sampler_name,
        scheduler,
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
        # Loop LoRAs
        lora_loop_1="",
        lora_strength_1=1.0,
        lora_loop_2="",
        lora_strength_2=1.0,
        lora_loop_3="",
        lora_strength_3=1.0,
        lora_loop_4="",
        lora_strength_4=1.0,
        lora_loop_5="",
        lora_strength_5=1.0,
        lora_loop_6="",
        lora_strength_6=1.0,
        lora_loop_7="",
        lora_strength_7=1.0,
        lora_loop_8="",
        lora_strength_8=1.0,
        lora_loop_9="",
        lora_strength_9=1.0,
        lora_loop_10="",
        lora_strength_10=1.0,
        # Loop Images (optionale "Schnitt"-Quellen)
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
        #Loop reference_image
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
        print("WAN VIDEO EXTENDER PRO (Memory Optimized + Loop Images)")
        print("=" * 60)

        # Collect loop-specific settings
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

        loop_loras = [
            (lora_loop_1, lora_strength_1, lora_strength_1),
            (lora_loop_2, lora_strength_2, lora_strength_2),
            (lora_loop_3, lora_strength_3, lora_strength_3),
            (lora_loop_4, lora_strength_4, lora_strength_4),
            (lora_loop_5, lora_strength_5, lora_strength_5),
            (lora_loop_6, lora_strength_6, lora_strength_6),
            (lora_loop_7, lora_strength_7, lora_strength_7),
            (lora_loop_8, lora_strength_8, lora_strength_8),
            (lora_loop_9, lora_strength_9, lora_strength_9),
            (lora_loop_10, lora_strength_10, lora_strength_10),
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

        # === TEMP DIRECTORY FOR SEGMENTS ===
        segment_dir = tempfile.mkdtemp(prefix="wan_segments_")
        print(f"🗂 Using temp segment dir: {segment_dir}")

        segment_paths = []  # generated segments (per loop)
        original_input_segment_path = None
        has_initial_input_from_user = (image is not None) or (video is not None)

        # === LOAD INITIAL INPUT (MEMORY OPTIMIZED) ===
        initial_frames = None
        if image is not None:
            print("Input: IMAGE")
            initial_frames = image
        elif video is not None:
            print("Input: VIDEO")
            if isinstance(video, torch.Tensor):
                initial_frames = video
            else:
                initial_frames = self._load_video_from_file(video)

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

            # Save full input as first segment for final combine (unverändert)
            original_input_segment_path = os.path.join(segment_dir, "segment_input.pt")
            try:
                torch.save(initial_frames, original_input_segment_path)
                print(f"💾 Saved original input segment: {original_input_segment_path}")
            except Exception as e:
                print(f"⚠ Failed to save original input segment: {e}")
                original_input_segment_path = None

            # Keep only last N frames in RAM as overlap context
            slice_len = min(num_input_frames, overlap_frames)
            if slice_len > 0:
                last_slice = initial_frames[-slice_len:]
                for i in range(slice_len):
                    frame = self._normalize_frame(last_slice[i])
                    context_frames.append(frame)
                H, W = context_frames[0].shape[0], context_frames[0].shape[1]

            # Free big input tensor from RAM
            del initial_frames
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        else:
            # No user input: create a single default frame only for first loop context
            print(f"⚠ No input - using default {default_height}x{default_width}")
            default_frame = torch.zeros(
                (default_height, default_width, 3), dtype=torch.float32
            )
            context_frames.append(default_frame)
            H, W = default_height, default_width

        # Base model/clip (unpatched)
        base_model = model
        base_clip = clip

        current_seed = seed

        # === MAIN LOOP (SEGMENT-BASED) ===
        for loop_idx in range(extension_loops):
            loop_id = loop_idx + 1

            # Aggressives Cleanup vor *jedem* Loop:
            try:
                comfy.model_management.unload_all_models()
            except Exception as e:
                print(f"⚠ unload_all_models() failed (ignored): {e}")

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

            # === LOAD LOOP-SPECIFIC LORA ===
            if loop_idx < len(loop_loras):
                lora_name, lora_str_model, lora_str_clip = loop_loras[loop_idx]
                if lora_name and str(lora_name).strip():
                    model, clip = self._load_lora(
                        base_model, base_clip, lora_name, lora_str_model, lora_str_clip
                    )
                else:
                    model = base_model
                    clip = base_clip
            else:
                model = base_model
                clip = base_clip

            # === SELECT PROMPT FOR THIS LOOP (Base + optional Loop-Override) ===
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

            # === CONDITIONING TEXT ===
            def get_cond(text, clip_obj):
                tokens = clip_obj.tokenize(text)
                cond, pooled = clip_obj.encode_from_tokens(tokens, return_pooled=True)
                return [[cond, {"pooled_output": pooled}]]

            c_pos = get_cond(active_prompt, clip)
            c_neg = get_cond(negative_prompt, clip)

            # === KONTEXT FÜR DIESEN LOOP BESTIMMEN ===
            loop_image = loop_images[loop_idx] if loop_idx < len(loop_images) else None

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

            # Auswahl der tatsächlich verwendeten Kontextframes (max overlap_frames, max generate_frames)
            if len(base_context_candidate) > 0:
                max_context = min(len(base_context_candidate), overlap_frames, generate_frames)
                selected_context_frames = base_context_candidate[-max_context:]
            else:
                selected_context_frames = []

            # === PREPARE BATCH ===
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

                write_context = True

                if loop_idx == 0:
                    # Im ersten Loop immer schreiben, wenn wir Frames haben (Startbild)
                    write_context = True 
                    print(f"🚀 Loop 1: Forcing Input Image injection ({context_count} frames)")
                elif loop_use_overlap[loop_idx]:
                    # In späteren Loops auf den User-Switch hören
                    write_context = True
                
                if write_context:
                    full_pixels[:context_count] = context_batch
                    full_masks[:context_count] = 0.0 # WICHTIG: Maske auf 0 (Fest), damit es bleibt!
                    print(f"✅ Context wrote to canvas: {context_count} frames")
                else:
                    context_count = 0
                    print("❌ Context/Overlap disabled for this loop")
                # --- KORREKTUR ENDE ---

        
            else:
                context_batch = None

            # === VACE LOGIC (mit Reference Image – unverändert) ===
            # override only if loop has its own ref
            current_loop_reference = reference_image
            
            if loop_reference_images[loop_idx] is not None:
                current_loop_reference = loop_reference_images[loop_idx]
                print(f"🎨 Loop {loop_id}: Overriding global reference with local loop image.")


            
            
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

            # Apply VACE
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

            # === EMPTY LATENT ===
            latent_length = ((generate_frames - 1) // 4) + 1 + trim_latent
            empty_latent = torch.zeros(
                [1, 16, latent_length, H // 8, W // 8],
                device=comfy.model_management.get_torch_device(),
            )
            latents = {"samples": empty_latent}

            print(f"Empty latent shape: {empty_latent.shape}")

            # === SAMPLING ===
            print("🎬 Sampling...")
            out = nodes.common_ksampler(
                model=model,
                seed=current_seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=new_pos,
                negative=new_neg,
                latent=latents,
                denoise=1.0,
            )
            new_samples = out[0]["samples"]

            # Trim reference frames if present
            if trim_latent > 0:
                new_samples = new_samples[:, :, trim_latent:, :, :]

            # === DECODE ===
            decoded = vae.decode(new_samples)
            decoded = decoded.cpu()

            if decoded.dim() == 5:
                decoded = decoded[0]

            print(f"✓ Decoded: {decoded.shape}")

            # Cleanup heavy tensors not needed anymore (VRAM)
            del new_samples, vace_latents, vace_masks, empty_latent, latents
            del c_pos, c_neg, new_pos, new_neg, context_batch
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # === ADD FRAMES AS DISK SEGMENT (ONLY NEW PART, Kontext überspringen) ===
            skip_count = context_count
            new_frames_list = []
            for j in range(skip_count, decoded.shape[0]):
                frame = self._normalize_frame(decoded[j]).cpu()
                new_frames_list.append(frame)

            new_frames = len(new_frames_list)
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
                # No new frames — keep context as is
                print(
                    "⚠ No new frames generated in this loop, keeping previous context_frames"
                )

            # Cleanup loop-local variables (CPU)
            del decoded, new_frames_list
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            current_seed += 1

        # === FINAL COMBINE FROM DISK SEGMENTS ===
        print("\n" + "=" * 60)
        print("COMPLETE - combining segments from disk")
        print("=" * 60 + "\n")

        final_tensors = []

        # 1) If user provided input (image or video), prepend original input segment
        if has_initial_input_from_user and original_input_segment_path is not None:
            try:
                base_tensor = torch.load(
                    original_input_segment_path, map_location="cpu"
                )
                final_tensors.append(base_tensor)
                print(
                    f"📥 Loaded original input segment ({base_tensor.shape[0]} frames)"
                )
            except Exception as e:
                print(f"⚠ Failed to load original input segment: {e}")

        # 2) Append all generated segments in order
        total_generated = 0
        for p in segment_paths:
            try:
                seg = torch.load(p, map_location="cpu")
                final_tensors.append(seg)
                total_generated += seg.shape[0]
                print(
                    f"📥 Loaded segment {os.path.basename(p)} ({seg.shape[0]} frames)"
                )
            except Exception as e:
                print(f"⚠ Failed to load segment {p}: {e}")

        # 3) If no user input, output besteht nur aus generierten Segmenten
        if len(final_tensors) > 0:
            full_video = torch.cat(final_tensors, dim=0)
        else:
            full_video = torch.zeros((1, default_height, default_width, 3))

        # Cleanup temp directory
        try:
            shutil.rmtree(segment_dir)
            print(f"🧹 Cleaned temp dir: {segment_dir}")
        except Exception as e:
            print(f"⚠ Could not remove temp dir {segment_dir}: {e}")

        return (full_video, "\n".join(used_prompts_log))


NODE_CLASS_MAPPINGS = {
    "WanVideoExtenderNative": WanVideoExtenderNative
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoExtenderNative": "Wan 2.2 Video Extender PRO"
}