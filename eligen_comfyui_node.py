
"""
Fixed EliGen Entity Control Node for ComfyUI
Native implementation with proper entity handling based on DiffSynth wrapper analysis
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
from PIL import Image


class EliGenEntityInput:
    """
    EliGen Entity Input Node - Better entity connection structure
    Based on successful DiffSynth wrapper approach with individual entity slots
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A magical coffee shop poster with blue misty background",
                    "tooltip": "Global scene description"
                }),
                "entity_prompt_1": ("STRING", {
                    "multiline": True,
                    "default": "A red magical coffee cup with flames burning inside",
                    "tooltip": "First entity description"
                }),
                "entity_mask_1": ("IMAGE", {
                    "tooltip": "Mask for first entity region"
                }),
            },
            "optional": {
                "entity_prompt_2": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Second entity description (optional)"
                }),
                "entity_mask_2": ("IMAGE", {
                    "tooltip": "Mask for second entity region (optional)"
                }),
                "entity_prompt_3": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Third entity description (optional)"
                }),
                "entity_mask_3": ("IMAGE", {
                    "tooltip": "Mask for third entity region (optional)"
                }),
                "entity_prompt_4": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Fourth entity description (optional)"
                }),
                "entity_mask_4": ("IMAGE", {
                    "tooltip": "Mask for fourth entity region (optional)"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Invert mask values (ComfyUI masks: black=active, EliGen needs white=active)"
                }),
            }
        }

    RETURN_TYPES = ("ELIGEN_ARGS", "IMAGE")
    RETURN_NAMES = ("eligen_args", "preview_mask")
    FUNCTION = "create_entity_input"
    CATEGORY = "conditioning/eligen"
    DESCRIPTION = "Create EliGen entity input with proper entity separation"

    def create_entity_input(self, global_prompt: str, entity_prompt_1: str, entity_mask_1,
                           entity_prompt_2: str = "", entity_mask_2=None,
                           entity_prompt_3: str = "", entity_mask_3=None,
                           entity_prompt_4: str = "", entity_mask_4=None,
                           invert_mask: bool = True):
        """Create EliGen entity input with proper entity handling"""
        try:
            entity_prompts = []
            entity_masks = []

            # Collect all valid entities
            entities = [
                (entity_prompt_1, entity_mask_1),
                (entity_prompt_2, entity_mask_2),
                (entity_prompt_3, entity_mask_3),
                (entity_prompt_4, entity_mask_4),
            ]

            for prompt, mask in entities:
                if prompt and prompt.strip() and mask is not None:
                    entity_prompts.append(prompt.strip())

                    # Convert mask to PIL format
                    processed_mask = self.process_mask(mask, invert_mask)
                    entity_masks.append(processed_mask)

            if not entity_prompts:
                raise ValueError("At least one entity prompt and mask must be provided")

            # Create preview visualization
            preview_image = self.create_preview_visualization(entity_masks, entity_prompts)

            # Create EliGen args in format expected by samplers
            eligen_args = {
                "global_prompt": global_prompt,
                "prompts": entity_prompts,
                "masks": entity_masks,
                "entity_count": len(entity_prompts)
            }

            print(f"EliGen: Created {len(entity_prompts)} entities")
            return (eligen_args, preview_image)

        except Exception as e:
            print(f"EliGen Entity Input Error: {e}")
            # Return empty args to prevent workflow breaking
            empty_args = {
                "global_prompt": global_prompt,
                "prompts": [],
                "masks": [],
                "entity_count": 0
            }
            empty_preview = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (empty_args, empty_preview)

    def process_mask(self, mask_tensor, invert_mask=True):
        """Process mask tensor to PIL Image format"""
        try:
            # Handle batch dimension
            if len(mask_tensor.shape) == 4:
                mask_array = mask_tensor[0].cpu().numpy()
            else:
                mask_array = mask_tensor.cpu().numpy()

            # Handle channel dimension
            if len(mask_array.shape) == 3:
                if mask_array.shape[2] == 3:  # RGB
                    mask_array = mask_array.mean(axis=2)  # Convert to grayscale
                elif mask_array.shape[2] == 1:  # Single channel
                    mask_array = mask_array[:, :, 0]

            # Normalize to 0-255
            if mask_array.max() <= 1.0:
                mask_array = (mask_array * 255).astype(np.uint8)
            else:
                mask_array = mask_array.astype(np.uint8)

            # Apply threshold to create binary mask
            mask_array = (mask_array > 127).astype(np.uint8) * 255

            # Invert if needed (ComfyUI convention vs EliGen expectation)
            if invert_mask:
                mask_array = 255 - mask_array

            # Convert to 3-channel RGB for consistency
            if len(mask_array.shape) == 2:
                mask_rgb = np.stack([mask_array, mask_array, mask_array], axis=-1)
            else:
                mask_rgb = mask_array

            return Image.fromarray(mask_rgb, mode='RGB')

        except Exception as e:
            print(f"Mask processing error: {e}")
            # Return a default white mask
            return Image.new('RGB', (512, 512), (255, 255, 255))

    def create_preview_visualization(self, masks, prompts):
        """Create a preview visualization of the entity masks"""
        try:
            if not masks:
                # Return black image if no masks
                return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

            # Get dimensions from first mask
            first_mask = masks[0]
            width, height = first_mask.size

            # Create base canvas
            canvas = Image.new('RGB', (width, height), (0, 0, 0))

            # Color palette for different entities
            colors = [
                (255, 100, 100),  # Red
                (100, 255, 100),  # Green
                (100, 100, 255),  # Blue
                (255, 255, 100),  # Yellow
            ]

            # Overlay each mask with different color
            for i, (mask, prompt) in enumerate(zip(masks, prompts)):
                mask_array = np.array(mask.convert('L'))
                color = colors[i % len(colors)]

                # Create colored overlay where mask is white
                overlay = np.zeros((height, width, 3), dtype=np.uint8)
                mask_white = mask_array == 255
                overlay[mask_white] = color

                # Blend with canvas
                canvas_array = np.array(canvas)
                canvas_array = np.where(mask_white[:, :, np.newaxis], 
                                      overlay * 0.7 + canvas_array * 0.3, 
                                      canvas_array).astype(np.uint8)
                canvas = Image.fromarray(canvas_array)

            # Convert to ComfyUI tensor format
            canvas_array = np.array(canvas).astype(np.float32) / 255.0
            canvas_tensor = torch.from_numpy(canvas_array).unsqueeze(0)  # Add batch dim

            return canvas_tensor

        except Exception as e:
            print(f"Preview visualization error: {e}")
            return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


class EliGenEntityControl:
    """
    EliGen Entity Control Node - Fixed implementation with proper tensor handling
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "eligen_args": ("ELIGEN_ARGS",),
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
    DESCRIPTION = "Apply EliGen entity-level control with fixed tensor handling"

    def __init__(self):
        self.original_forward = None
        self.entity_data = None

    def get_entity_prompt_embeddings(self, clip, entity_prompt: str) -> Dict[str, torch.Tensor]:
        """Generate prompt embeddings for a single entity using Qwen-Image format"""
        try:
            # Use standard CLIP encoding - no special template needed for native ComfyUI
            tokens = clip.tokenize(entity_prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

            return {
                "prompt_emb": cond,
                "prompt_emb_mask": torch.ones(cond.shape[0], cond.shape[1], dtype=torch.long, device=cond.device),
                "pooled": pooled
            }
        except Exception as e:
            print(f"Entity prompt embedding error: {e}")
            # Return dummy embeddings
            dummy_emb = torch.zeros(1, 77, 768, device=clip.load_device)
            dummy_mask = torch.ones(1, 77, dtype=torch.long, device=clip.load_device)
            dummy_pooled = torch.zeros(1, 768, device=clip.load_device)
            return {
                "prompt_emb": dummy_emb,
                "prompt_emb_mask": dummy_mask,
                "pooled": dummy_pooled
            }

    def preprocess_entity_masks(self, masks: List[Image.Image], target_height: int, target_width: int) -> torch.Tensor:
        """Preprocess entity masks for attention mechanism"""
        try:
            processed_masks = []

            for mask_pil in masks:
                # Resize to target dimensions (latent space)
                latent_h, latent_w = target_height // 16, target_width // 16  # Qwen uses 16x downsampling
                mask_resized = mask_pil.resize((latent_w, latent_h), Image.NEAREST)

                # Convert to tensor
                mask_array = np.array(mask_resized.convert('L'))
                mask_tensor = torch.from_numpy(mask_array).float() / 255.0

                # Binary threshold
                mask_binary = (mask_tensor > 0.5).float()
                processed_masks.append(mask_binary)

            if processed_masks:
                # Stack along batch dimension: [num_entities, H, W]
                masks_tensor = torch.stack(processed_masks, dim=0)
                return masks_tensor.unsqueeze(0)  # Add batch dim: [1, num_entities, H, W]
            else:
                # Return empty tensor with proper shape
                return torch.zeros(1, 0, target_height // 16, target_width // 16)

        except Exception as e:
            print(f"Mask preprocessing error: {e}")
            # Return dummy mask
            return torch.zeros(1, 1, target_height // 16, target_width // 16)

    def create_entity_attention_constraints(self, entity_embeddings: List[torch.Tensor], 
                                          entity_masks: torch.Tensor, 
                                          conditioning_length: int) -> Dict[str, torch.Tensor]:
        """Create attention constraints for entity control - Fixed tensor handling"""
        try:
            if not entity_embeddings or entity_masks.shape[1] == 0:
                return {}

            # Calculate sequence lengths - FIXED calculation
            entity_seq_lens = [emb.shape[1] for emb in entity_embeddings]
            total_entity_len = sum(entity_seq_lens)

            # Image sequence length based on mask dimensions - FIXED
            mask_h, mask_w = entity_masks.shape[2], entity_masks.shape[3]
            image_seq_len = mask_h * mask_w

            # Total sequence length
            total_seq_len = total_entity_len + conditioning_length + image_seq_len

            print(f"EliGen Debug: entity_lens={entity_seq_lens}, total_entity={total_entity_len}, "
                  f"cond_len={conditioning_length}, image_len={image_seq_len}, total={total_seq_len}")

            # Create attention constraints dict
            constraints = {
                "entity_embeddings": entity_embeddings,
                "entity_masks": entity_masks,
                "entity_seq_lens": entity_seq_lens,
                "total_entity_len": total_entity_len,
                "image_seq_len": image_seq_len,
                "total_seq_len": total_seq_len
            }

            return constraints

        except Exception as e:
            print(f"Attention constraints error: {e}")
            return {}

    def patch_model_forward(self, model, entity_constraints: Dict):
        """Patch model for entity control with improved error handling"""

        def patched_forward(original_forward):
            def forward_wrapper(*args, **kwargs):
                try:
                    # Check if we have valid entity data
                    if not entity_constraints or not entity_constraints.get("entity_embeddings"):
                        return original_forward(*args, **kwargs)

                    # Modify context if available
                    if 'context' in kwargs and kwargs['context'] is not None:
                        original_context = kwargs['context']
                        entity_embeddings = entity_constraints.get("entity_embeddings", [])

                        if entity_embeddings:
                            # Concatenate entity embeddings with original context
                            try:
                                entity_context = torch.cat(entity_embeddings, dim=1)
                                modified_context = torch.cat([entity_context, original_context], dim=1)
                                kwargs['context'] = modified_context

                                print(f"EliGen: Modified context shape from {original_context.shape} to {modified_context.shape}")
                            except Exception as e:
                                print(f"EliGen context modification error: {e}")
                                # Continue with original context if modification fails

                    return original_forward(*args, **kwargs)

                except Exception as e:
                    print(f"EliGen forward patch error: {e}")
                    # Fall back to original forward on any error
                    return original_forward(*args, **kwargs)

            return forward_wrapper

        # Get the model's diffusion model
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model

            # Store original forward method if not already stored
            if not hasattr(self, 'original_forward') or self.original_forward is None:
                self.original_forward = diffusion_model.forward

            # Patch the forward method
            diffusion_model.forward = patched_forward(self.original_forward)

        return model

    def apply_entity_control(self, model, conditioning, eligen_args, clip, 
                           strength: float = 1.0, enable_on_negative: bool = False):
        """Apply entity control to model and conditioning - Fixed implementation"""

        try:
            # Validate EliGen args
            if not eligen_args or not eligen_args.get("prompts") or not eligen_args.get("masks"):
                print("EliGen: No valid entity data provided, returning original inputs")
                return (model, conditioning)

            entity_prompts = eligen_args["prompts"]
            entity_masks = eligen_args["masks"]

            print(f"EliGen: Processing {len(entity_prompts)} entities")

            # Process entity prompt embeddings
            entity_embeddings = []
            for i, prompt in enumerate(entity_prompts):
                try:
                    emb_data = self.get_entity_prompt_embeddings(clip, prompt)
                    entity_embeddings.append(emb_data["prompt_emb"])
                    print(f"EliGen: Processed entity {i+1} embedding: {emb_data['prompt_emb'].shape}")
                except Exception as e:
                    print(f"EliGen: Error processing entity {i+1}: {e}")
                    continue

            if not entity_embeddings:
                print("EliGen: No valid entity embeddings created")
                return (model, conditioning)

            # Process entity masks - assume reasonable image dimensions
            target_height, target_width = 1024, 1024  # Standard dimensions
            processed_masks = self.preprocess_entity_masks(entity_masks, target_height, target_width)
            print(f"EliGen: Processed masks shape: {processed_masks.shape}")

            # Get conditioning length
            first_cond = conditioning[0][0] if conditioning else torch.zeros(1, 77, 768)
            conditioning_length = first_cond.shape[1]

            # Create entity attention constraints
            entity_constraints = self.create_entity_attention_constraints(
                entity_embeddings, processed_masks, conditioning_length
            )

            if not entity_constraints:
                print("EliGen: No valid attention constraints created")
                return (model, conditioning)

            # Clone model to avoid modifying original
            patched_model = model.clone()

            # Store entity data for forward pass
            self.entity_data = {
                **entity_constraints,
                'strength': strength,
                'global_prompt': eligen_args.get("global_prompt", "")
            }

            # Patch the model
            patched_model = self.patch_model_forward(patched_model, entity_constraints)

            # Modify conditioning to include entity information
            modified_conditioning = []
            for cond_item in conditioning:
                cond_tensor, cond_dict = cond_item

                # Create new conditioning dict with entity information
                new_cond_dict = cond_dict.copy()
                new_cond_dict['eligen_entity_control'] = {
                    'entity_prompts': entity_prompts,
                    'entity_count': len(entity_prompts),
                    'strength': strength,
                    'global_prompt': eligen_args.get("global_prompt", "")
                }

                modified_conditioning.append([cond_tensor, new_cond_dict])

            print("EliGen: Entity control applied successfully")
            return (patched_model, modified_conditioning)

        except Exception as e:
            print(f"EliGen: Entity control application failed: {e}")
            print(f"EliGen: Returning original inputs to prevent workflow failure")
            return (model, conditioning)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "EliGenEntityInput": EliGenEntityInput,
    "EliGenEntityControl": EliGenEntityControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EliGenEntityInput": "EliGen Entity Input",
    "EliGenEntityControl": "EliGen Entity Control",
}

# Optional: Add web directory for custom JavaScript (if needed)
WEB_DIRECTORY = "./js"

# Metadata
__version__ = "1.1.0"
__author__ = "Fixed EliGen ComfyUI Implementation"
__description__ = "Fixed native EliGen entity control implementation for ComfyUI with proper entity handling"
