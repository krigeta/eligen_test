# ComfyUI/custom_nodes/ComfyUI-NativeEliGen/__init__.py

import torch
import torch.nn.functional as F
from typing import List, Optional
import comfy.utils
import comfy.model_management
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
import logging

logger = logging.getLogger(__name__)

class NativeEliGenConditioner:
    """
    Fully native ComfyUI node for EliGen regional attention.
    Accepts lists of CONDITIONING and MASK tensors.
    Patches and outputs MODEL for KSampler compatibility.
    Zero string parsing. Pure tensor I/O.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "global_conditioning": ("CONDITIONING",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "entity_conds": ("CONDITIONING", {"tooltip": "List of entity conditioning from CLIPTextEncode"}),
                "entity_masks": ("MASK", {"tooltip": "List of masks from ControlNet, MaskEditor, etc."}),
                "negative_conditioning": ("CONDITIONING",),
                "enable_negative_eligen": ("BOOLEAN", {"default": False}),
                "enable_inpaint": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply_eligen"
    CATEGORY = "conditioning/EliGen"

    def apply_eligen(self,
                    model: ModelPatcher,
                    clip: CLIP,
                    global_conditioning,
                    width: int,
                    height: int,
                    entity_conds = None,
                    entity_masks = None,
                    negative_conditioning = None,
                    enable_negative_eligen: bool = False,
                    enable_inpaint: bool = False):

        device = comfy.model_management.get_torch_device()

        # Extract global conditioning
        cond_vec, cond_dict = global_conditioning[0]
        cond_vec = cond_vec.to(device)
        pooled = cond_dict.get("pooled_output", None)
        if pooled is not None:
            pooled = pooled.to(device)

        # Extract negative conditioning
        if negative_conditioning is None:
            neg_tokens = clip.tokenize("")
            neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
            neg_cond = neg_cond.to(device)
            neg_pooled = neg_pooled.to(device) if neg_pooled is not None else None
        else:
            neg_vec, neg_dict = negative_conditioning[0]
            neg_cond = neg_vec.to(device)
            neg_pooled = neg_dict.get("pooled_output", None)
            if neg_pooled is not None:
                neg_pooled = neg_pooled.to(device)

        # Prepare conditioning dicts (for output)
        positive = [[cond_vec, {
            "pooled_output": pooled,
            "width": width,
            "height": height,
        }]]

        negative = [[neg_cond, {
            "pooled_output": neg_pooled,
            "width": width,
            "height": height,
        }]]

        # If no entities, return original model
        if entity_conds is None or entity_masks is None:
            return (model, positive, negative)

        # Normalize to lists
        if not isinstance(entity_conds, list):
            entity_conds = [entity_conds]
        if not isinstance(entity_masks, list):
            entity_masks = [entity_masks]

        if len(entity_conds) != len(entity_masks):
            raise ValueError(f"Entity count mismatch: {len(entity_conds)} conds vs {len(entity_masks)} masks")

        # Extract conditioning vectors
        entity_cond_vectors = []
        for ec in entity_conds:
            if isinstance(ec, list):
                e_vec, _ = ec[0]
            else:
                e_vec, _ = ec
            entity_cond_vectors.append(e_vec.to(device))

        # Process masks → latent space + soft falloff
        latent_width = width // 8
        latent_height = height // 8
        processed_masks = []

        for mask in entity_masks:
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            mask = mask.to(device)

            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)

            if mask.shape[-2:] != (latent_height, latent_width):
                mask = F.interpolate(mask, size=(latent_height, latent_width), mode='bilinear', align_corners=False)

            mask = self._apply_soft_falloff(mask, kernel_size=5, sigma=1.0)
            processed_masks.append(mask)

        # Clone model and patch it
        model = model.clone()

        # Store EliGen data in model_options
        model.model_options["eligen_entity_conds"] = entity_cond_vectors
        model.model_options["eligen_entity_masks"] = processed_masks
        model.model_options["eligen_enable_on_negative"] = enable_negative_eligen
        model.model_options["eligen_enable_inpaint"] = enable_inpaint

        # Patch attention forward
        self._patch_model_forward(model)

        return (model, positive, negative)

    def _apply_soft_falloff(self, mask: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        if kernel_size % 2 == 0:
            kernel_size += 1

        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=mask.device)
        xx = ax.repeat(kernel_size, 1)
        yy = xx.t()
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        mask_padded = F.pad(mask, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate')
        mask_blurred = F.conv2d(mask_padded, kernel, padding=0, groups=mask.shape[1])

        return mask_blurred

    def _patch_model_forward(self, model: ModelPatcher):
        """Patch the model's attention layers to implement EliGen regional conditioning."""

        def eligen_attn_patch(h, transformer_options):
            cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
            is_cond = cond_or_uncond[0] == 0

            model_options = transformer_options.get("model_options", {})
            if not is_cond and not model_options.get("eligen_enable_on_negative", False):
                return h

            entity_conds = model_options.get("eligen_entity_conds", [])
            entity_masks = model_options.get("eligen_entity_masks", [])

            if not entity_conds or not entity_masks:
                return h

            context = transformer_options.get("context", None)
            if context is None:
                return h

            B, L, C = h.shape
            H = W = int(L ** 0.5)

            for i, (e_cond, e_mask) in enumerate(zip(entity_conds, entity_masks)):
                if e_cond is None or e_mask is None:
                    continue

                if e_mask.shape[-2:] != (H, W):
                    e_mask = F.interpolate(e_mask, size=(H, W), mode='bilinear', align_corners=False)

                if e_mask.shape[0] == 1 and B > 1:
                    e_mask = e_mask.expand(B, -1, -1, -1)
                elif e_mask.shape[0] != B:
                    e_mask = e_mask[:B] if e_mask.shape[0] > B else e_mask.expand(B, -1, -1, -1)

                mask_flat = e_mask.view(B, 1, H * W).permute(0, 2, 1)  # [B, L, 1]
                masked_context = e_cond * mask_flat
                context = torch.cat([context, masked_context], dim=1)

            transformer_options["context"] = context
            return h

        # Apply patch to all transformer blocks
        for k in model.model.diffusion_model.state_dict().keys():
            if ".attn1." in k and k.endswith(".weight"):
                layer_name = k.split(".attn1.")[0] + ".attn1"
                model.set_model_attn1_patch(eligen_attn_patch, layer_name)

        # Also patch top-level if needed (for FLUX/Qwen global attention layers)
        model.set_model_attn1_patch(eligen_attn_patch)


# --- NODE REGISTRATION ---
NODE_CLASS_MAPPINGS = {
    "NativeEliGenConditioner": NativeEliGenConditioner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NativeEliGenConditioner": "Native EliGen (Model + Cond Output)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
