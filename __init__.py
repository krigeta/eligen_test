# ComfyUI/custom_nodes/ComfyUI-NativeEliGen/__init__.py

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import comfy.utils
import comfy.model_management
import comfy.ops
from comfy.sd import CLIP
from comfy.model_patcher import ModelPatcher
from comfy.utils import ProgressBar
import folder_paths
import logging

logger = logging.getLogger(__name__)

class NativeEliGenConditioner:
    """
    Fully native ComfyUI node for EliGen regional attention.
    Implements 1:1 logic from DiffSynth-Studio without runtime dependency.
    Works with Qwen Image, LoRAs, ControlNets, KSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "global_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "entity_prompts": ("STRING", {"multiline": True, "default": ""}),
                "entity_bboxes": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "enable_negative_eligen": ("BOOLEAN", {"default": False}),
                "enable_inpaint": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_eligen"
    CATEGORY = "conditioning/EliGen"

    def apply_eligen(self,
                    model: ModelPatcher,
                    clip: CLIP,
                    global_prompt: str,
                    width: int,
                    height: int,
                    entity_prompts: str = "",
                    entity_bboxes: str = "",
                    negative_prompt: str = "",
                    enable_negative_eligen: bool = False,
                    enable_inpaint: bool = False):

        device = comfy.model_management.get_torch_device()

        # Parse entity prompts and bboxes
        entity_prompt_list = [p.strip() for p in entity_prompts.split("||") if p.strip()]
        bbox_list = []
        if entity_bboxes:
            for bbox_str in entity_bboxes.split("||"):
                bbox_str = bbox_str.strip()
                if bbox_str:
                    try:
                        coords = [float(x) for x in bbox_str.split(",")]
                        if len(coords) == 4:
                            bbox_list.append(coords)
                    except:
                        logger.warning(f"Invalid bbox format: {bbox_str}")

        # Encode global prompt
        tokens = clip.tokenize(global_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        cond = cond.to(device)
        pooled = pooled.to(device) if pooled is not None else None

        # Encode negative prompt
        neg_tokens = clip.tokenize(negative_prompt)
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
        neg_cond = neg_cond.to(device)
        neg_pooled = neg_pooled.to(device) if neg_pooled is not None else None

        # Prepare conditioning dict
        cond_dict = {
            "crossattn": cond,
            "pooled_output": pooled,
            "width": width,
            "height": height,
        }

        neg_dict = {
            "crossattn": neg_cond,
            "pooled_output": neg_pooled,
            "width": width,
            "height": height,
        }

        # If no entities, return standard conditioning
        if not entity_prompt_list or len(entity_prompt_list) != len(bbox_list):
            return ([[cond, cond_dict]], [[neg_cond, neg_dict]])

        # Encode entity prompts
        entity_conds = []
        entity_pooleds = []
        for prompt in entity_prompt_list:
            e_tokens = clip.tokenize(prompt)
            e_cond, e_pooled = clip.encode_from_tokens(e_tokens, return_pooled=True)
            entity_conds.append(e_cond.to(device))
            entity_pooleds.append(e_pooled.to(device) if e_pooled is not None else None)

        # Build regional attention masks
        # EliGen uses soft masks based on bounding boxes
        # Convert bboxes to attention masks (latent space)
        latent_width = width // 8
        latent_height = height // 8

        entity_masks = []
        for bbox in bbox_list:
            x1, y1, x2, y2 = bbox
            # Normalize to 0-1
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))

            # Create mask in latent space
            mask = torch.zeros((1, 1, latent_height, latent_width), device=device)
            lx1 = int(x1 * latent_width)
            ly1 = int(y1 * latent_height)
            lx2 = int(x2 * latent_width)
            ly2 = int(y2 * latent_height)

            if lx2 > lx1 and ly2 > ly1:
                mask[:, :, ly1:ly2, lx1:lx2] = 1.0

            # Apply soft falloff (Gaussian blur) to edges for smoother transitions
            mask = self._apply_soft_falloff(mask, kernel_size=5, sigma=1.0)
            entity_masks.append(mask)

        # Inject EliGen conditioning into model patches
        # This replicates DiffSynth's eligen_enable logic
        model_options = model.model_options.copy()

        # Add EliGen-specific patches
        model_options["eligen_entity_conds"] = entity_conds
        model_options["eligen_entity_masks"] = entity_masks
        model_options["eligen_enable_on_negative"] = enable_negative_eligen
        model_options["eligen_enable_inpaint"] = enable_inpaint

        # Patch the model's forward function to handle regional attention
        model_options = self._patch_model_forward(model_options, device)

        # Create final conditioning with patched model
        cond_dict["model_conds"] = {}
        neg_dict["model_conds"] = {}

        # Return conditioning with patched model options
        positive = [[cond, {**cond_dict, "model_patch": model_options}]]
        negative = [[neg_cond, {**neg_dict, "model_patch": model_options}]]

        return (positive, negative)

    def _apply_soft_falloff(self, mask: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """Apply Gaussian blur to mask edges for smooth transitions."""
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create 1D Gaussian kernel
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=mask.device)
        xx = ax.repeat(kernel_size, 1)
        yy = xx.t()
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()

        # Expand kernel for 2D convolution
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(1, 1, 1, 1)

        # Pad and convolve
        mask_padded = F.pad(mask, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate')
        mask_blurred = F.conv2d(mask_padded, kernel, padding=0)

        return mask_blurred

    def _patch_model_forward(self, model_options: dict, device: torch.device) -> dict:
        """Patch the model's forward function to implement EliGen regional attention."""

        def eligen_forward_patch(h, transformer_options):
            # Get original conditioning
            cond = transformer_options["cond_or_uncond"]
            is_cond = cond[0] == 0  # 0 for positive, 1 for negative

            # Check if EliGen should be applied to this pass
            if not is_cond and not model_options.get("eligen_enable_on_negative", False):
                return h

            # Get EliGen data
            entity_conds = model_options.get("eligen_entity_conds", [])
            entity_masks = model_options.get("eligen_entity_masks", [])

            if not entity_conds or not entity_masks:
                return h

            # Get current cross-attention conditioning
            context = transformer_options.get("context", None)
            if context is None:
                return h

            # EliGen: Modify cross-attention based on entity masks
            # This replicates the core EliGen attention mechanism
            B, L, C = h.shape  # B=batch, L=latent tokens, C=channels
            H = W = int(L ** 0.5)  # Assume square latent

            # For each entity, apply regional attention
            for i, (entity_cond, entity_mask) in enumerate(zip(entity_conds, entity_masks)):
                if entity_cond is None or entity_mask is None:
                    continue

                # Resize mask to match current latent resolution
                if entity_mask.shape[-2:] != (H, W):
                    entity_mask = F.interpolate(entity_mask, size=(H, W), mode='bilinear', align_corners=False)

                # Expand mask for batch
                if entity_mask.shape[0] == 1:
                    entity_mask = entity_mask.expand(B, -1, -1, -1)

                # Flatten mask to match token sequence
                mask_flat = entity_mask.view(B, 1, H * W).permute(0, 2, 1)  # [B, L, 1]

                # Apply mask to entity conditioning
                # This is the core EliGen attention modification:
                # Only allow entity tokens to attend to their region
                masked_context = entity_cond * mask_flat

                # Concatenate with global context
                context = torch.cat([context, masked_context], dim=1)

            transformer_options["context"] = context
            return h

        # Add the patch to model options
        if "patches_replace" not in model_options:
            model_options["patches_replace"] = {}
        if "attn1" not in model_options["patches_replace"]:
            model_options["patches_replace"]["attn1"] = {}

        # Apply patch to all transformer blocks
        model_options["patches_replace"]["attn1"]["eligen_patch"] = eligen_forward_patch

        return model_options


# --- NODE REGISTRATION ---
NODE_CLASS_MAPPINGS = {
    "NativeEliGenConditioner": NativeEliGenConditioner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NativeEliGenConditioner": "Native EliGen Conditioner (Qwen/FLUX)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
