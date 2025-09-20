
"""
EliGen Entity Control Node for ComfyUI
Implements EliGen inference logic natively without DiffSynth dependency
Compatible with existing ComfyUI Qwen image nodes and workflows
"""

import torch
import torch.nn.functional as F
import numpy as np
import comfy.model_management
import comfy.utils
import comfy.model_patcher
import node_helpers
import math
from typing import List, Tuple, Optional, Dict, Any


class EliGenEntityControl:
    """
    EliGen Entity Control Node - Implements entity-level control for Qwen image generation
    Replicates DiffSynth-Studio EliGen functionality natively in ComfyUI
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "entity_prompts": ("STRING", {
                    "multiline": True, 
                    "default": "entity1, entity2, entity3",
                    "tooltip": "Entity prompts separated by commas"
                }),
                "entity_masks": ("IMAGE",),  # Batch of masks for each entity
                "clip": ("CLIP",),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1,
                    "tooltip": "Control strength for entity effects"
                }),
                "enable_on_negative": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply entity control to negative conditioning"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "apply_entity_control"
    CATEGORY = "conditioning/eligen"
    DESCRIPTION = "Apply EliGen entity-level control to Qwen image generation"

    def __init__(self):
        self.original_forward = None
        self.entity_data = None

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        """Extract hidden states based on attention mask"""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def get_entity_prompt_embeddings(self, clip, entity_prompt: str) -> Dict[str, torch.Tensor]:
        """Generate prompt embeddings for a single entity"""
        # Use Qwen-Image template format
        template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 34

        formatted_text = template.format(entity_prompt)
        tokens = clip.tokenize(formatted_text)

        # Get embeddings
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        # Drop template tokens (similar to DiffSynth implementation)
        if cond.shape[1] > drop_idx:
            cond = cond[:, drop_idx:]

        return {
            "prompt_emb": cond,
            "prompt_emb_mask": torch.ones(cond.shape[0], cond.shape[1], dtype=torch.long, device=cond.device),
            "pooled": pooled
        }

    def preprocess_masks(self, masks: torch.Tensor, height: int, width: int) -> List[torch.Tensor]:
        """Preprocess entity masks for regional attention"""
        # Convert from ComfyUI image format (BHWC) to mask format
        if len(masks.shape) == 4:
            # Take first channel if RGB, convert to grayscale
            masks = masks.mean(dim=-1, keepdim=True)  # BHWC -> BHW1

        processed_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i].squeeze(-1)  # HW

            # Resize to latent dimensions
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), 
                size=(height // 8, width // 8), 
                mode='nearest'
            )

            # Convert to binary mask
            mask = (mask > 0.5).float()
            processed_masks.append(mask.squeeze(0))  # Remove batch dim

        return processed_masks

    def prepare_entity_inputs(self, clip, entity_prompts: List[str], entity_masks: torch.Tensor, 
                            height: int, width: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Prepare entity inputs similar to DiffSynth implementation"""

        # Process masks
        processed_masks = self.preprocess_masks(entity_masks, height, width)
        entity_masks_tensor = torch.stack(processed_masks, dim=0).unsqueeze(0)  # 1, N, C, H, W

        # Process prompts
        entity_prompt_embs = []
        entity_prompt_masks = []

        for entity_prompt in entity_prompts:
            emb_data = self.get_entity_prompt_embeddings(clip, entity_prompt.strip())
            entity_prompt_embs.append(emb_data["prompt_emb"])
            entity_prompt_masks.append(emb_data["prompt_emb_mask"])

        return entity_prompt_embs, entity_prompt_masks, entity_masks_tensor

    def create_entity_attention_mask(self, entity_prompt_embs: List[torch.Tensor], 
                                   entity_masks: torch.Tensor, image_seq_len: int, 
                                   global_prompt_len: int) -> torch.Tensor:
        """Create attention mask for regional entity control"""
        batch_size = 1
        entity_seq_lens = [emb.shape[1] for emb in entity_prompt_embs]
        total_entity_len = sum(entity_seq_lens)
        total_seq_len = total_entity_len + global_prompt_len + image_seq_len

        # Initialize attention mask (True = allow attention, False = mask)
        attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), dtype=torch.bool, device=entity_masks.device)

        # Apply entity-specific regional constraints
        cumsum = [0]
        for length in entity_seq_lens:
            cumsum.append(cumsum[-1] + length)

        global_start = total_entity_len
        global_end = global_start + global_prompt_len
        image_start = global_end
        image_end = total_seq_len

        # Process each entity mask
        for i, (start_idx, end_idx) in enumerate(zip(cumsum[:-1], cumsum[1:])):
            if i < entity_masks.shape[1]:  # Ensure we have a mask for this entity
                # Get entity mask and reshape for image tokens
                entity_mask = entity_masks[0, i, 0]  # H, W

                # Create image-level mask
                flat_mask = entity_mask.flatten()
                image_mask = flat_mask.unsqueeze(0).repeat(entity_seq_lens[i], 1)  # entity_len, image_len

                # Apply regional attention: entity tokens only attend to their regions
                attention_mask[:, start_idx:end_idx, image_start:image_end] = image_mask
                attention_mask[:, image_start:image_end, start_idx:end_idx] = image_mask.transpose(0, 1)

        # Block cross-entity attention (entities don't attend to each other)
        for i in range(len(entity_seq_lens)):
            for j in range(len(entity_seq_lens)):
                if i != j:
                    start_i, end_i = cumsum[i], cumsum[i+1]
                    start_j, end_j = cumsum[j], cumsum[j+1]
                    attention_mask[:, start_i:end_i, start_j:end_j] = False

        # Convert to attention bias (0 = attend, -inf = don't attend)
        attention_bias = torch.zeros_like(attention_mask.float())
        attention_bias[~attention_mask] = float('-inf')

        return attention_bias

    def patch_qwen_model_forward(self, model, entity_data: Dict):
        """Patch Qwen model to support entity control during forward pass"""

        def patched_forward(original_forward):
            def forward_wrapper(*args, **kwargs):
                # Check if we have entity data for this forward pass
                if not hasattr(self, 'entity_data') or self.entity_data is None:
                    return original_forward(*args, **kwargs)

                # Extract relevant arguments
                if 'context' in kwargs:
                    context = kwargs['context']

                    # Modify context to include entity information
                    if context is not None and self.entity_data:
                        # Concatenate entity embeddings with global context
                        entity_embs = self.entity_data.get('entity_prompt_embs', [])
                        if entity_embs:
                            entity_context = torch.cat(entity_embs, dim=1)  # Concat along sequence dim
                            modified_context = torch.cat([entity_context, context], dim=1)
                            kwargs['context'] = modified_context

                            # Add attention mask if available
                            if 'attention_mask' not in kwargs and 'entity_attention_mask' in self.entity_data:
                                kwargs['attention_mask'] = self.entity_data['entity_attention_mask']

                return original_forward(*args, **kwargs)

            return forward_wrapper

        # Get the model's diffusion model
        diffusion_model = model.model.diffusion_model

        # Store original forward method
        if not hasattr(self, 'original_forward') or self.original_forward is None:
            self.original_forward = diffusion_model.forward

        # Patch the forward method
        diffusion_model.forward = patched_forward(self.original_forward)

        return model

    def apply_entity_control(self, model, conditioning, entity_prompts: str, entity_masks, clip, 
                           strength: float = 1.0, enable_on_negative: bool = False):
        """Main function to apply entity control to model and conditioning"""

        # Parse entity prompts
        entity_prompt_list = [p.strip() for p in entity_prompts.split(',') if p.strip()]

        if not entity_prompt_list:
            return (model, conditioning)

        # Ensure we have enough masks
        if entity_masks.shape[0] < len(entity_prompt_list):
            print(f"Warning: Only {entity_masks.shape[0]} masks provided for {len(entity_prompt_list)} entities")
            entity_prompt_list = entity_prompt_list[:entity_masks.shape[0]]

        # Get dimensions from first conditioning
        if not conditioning:
            return (model, conditioning)

        first_cond = conditioning[0][0]
        # Assume standard latent dimensions (height and width will be inferred)
        height, width = 1024, 1024  # Default dimensions, can be made configurable

        # Prepare entity inputs
        entity_prompt_embs, entity_prompt_masks, processed_entity_masks = self.prepare_entity_inputs(
            clip, entity_prompt_list, entity_masks, height, width
        )

        # Clone model to avoid modifying original
        patched_model = model.clone()

        # Store entity data for use in patched forward
        self.entity_data = {
            'entity_prompt_embs': entity_prompt_embs,
            'entity_prompt_masks': entity_prompt_masks,
            'entity_masks': processed_entity_masks,
            'strength': strength
        }

        # Calculate attention mask dimensions
        image_seq_len = (height // 16) * (width // 16)  # Qwen uses 16x downsampling
        global_prompt_len = first_cond.shape[1]

        # Create entity attention mask
        entity_attention_mask = self.create_entity_attention_mask(
            entity_prompt_embs, processed_entity_masks, image_seq_len, global_prompt_len
        )
        self.entity_data['entity_attention_mask'] = entity_attention_mask

        # Patch the model
        patched_model = self.patch_qwen_model_forward(patched_model, self.entity_data)

        # Modify conditioning to include entity information
        modified_conditioning = []
        for cond_item in conditioning:
            cond_tensor, cond_dict = cond_item

            # Create new conditioning dict with entity information
            new_cond_dict = cond_dict.copy()
            new_cond_dict['eligen_entity_control'] = {
                'entity_prompts': entity_prompt_list,
                'entity_masks': processed_entity_masks,
                'entity_embeddings': entity_prompt_embs,
                'strength': strength,
                'enable_on_negative': enable_on_negative
            }

            modified_conditioning.append([cond_tensor, new_cond_dict])

        return (patched_model, modified_conditioning)


class EliGenMaskPreprocessor:
    """
    Helper node to preprocess masks for EliGen entity control
    Converts images to proper mask format
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of mask images
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "preprocess_masks"
    CATEGORY = "conditioning/eligen"

    def preprocess_masks(self, images, threshold=0.5, invert=False):
        """Convert images to binary masks suitable for EliGen"""

        # Convert to grayscale if needed
        if images.shape[-1] == 3:  # RGB
            masks = images.mean(dim=-1, keepdim=True)  # Convert to grayscale
        else:
            masks = images

        # Apply threshold
        masks = (masks > threshold).float()

        # Invert if requested
        if invert:
            masks = 1.0 - masks

        return (masks,)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "EliGenEntityControl": EliGenEntityControl,
    "EliGenMaskPreprocessor": EliGenMaskPreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EliGenEntityControl": "EliGen Entity Control",
    "EliGenMaskPreprocessor": "EliGen Mask Preprocessor",
}

# Optional: Add web directory for custom JavaScript (if needed)
WEB_DIRECTORY = "./js"

# Metadata
__version__ = "1.0.0"
__author__ = "EliGen ComfyUI Implementation"
__description__ = "Native EliGen entity control implementation for ComfyUI"
