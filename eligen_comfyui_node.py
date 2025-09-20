"""
FIXED QWEN ELIGEN CUSTOM NODE FOR COMFYUI
Fixes: 1) Mask preview showing all entities, 2) Proper regional entity control
Version: 2.1.0 (Bug Fixes Applied)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
from typing import List, Dict, Any, Optional, Tuple
from einops import rearrange
import logging
import os
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedProcessEntityMasksExtension:
    """
    FIXED Extension class - corrected attention masking for proper regional control
    """
    
    @staticmethod
    def add_process_entity_masks_to_model(model):
        """
        Add FIXED process_entity_masks method with proper regional attention
        """
        
        def process_entity_masks(self, latents, prompt_emb, prompt_emb_mask, 
                                entity_prompt_emb, entity_prompt_emb_mask, 
                                entity_masks, height, width, image, img_shapes):
            """
            FIXED process_entity_masks with proper regional attention constraints
            """
            try:
                logger.info(f"FIXED: Processing {len(entity_prompt_emb)} entities for regional control")
                
                # 1. Process and concatenate embeddings
                processed_embeddings = []
                for i, local_prompt_emb in enumerate(entity_prompt_emb + [prompt_emb]):
                    if hasattr(self, 'txt_norm'):
                        normed = self.txt_norm(local_prompt_emb)
                    else:
                        normed = local_prompt_emb
                    
                    if hasattr(self, 'txt_in'):
                        processed = self.txt_in(normed)
                    else:
                        processed = normed
                    
                    processed_embeddings.append(processed)
                    logger.debug(f"FIXED: Processed entity {i+1} embedding: {processed.shape}")
                
                # Concatenate all processed embeddings
                all_prompt_emb = torch.cat(processed_embeddings, dim=1)
                logger.info(f"FIXED: Combined embeddings shape: {all_prompt_emb.shape}")
                
                # 2. Create proper spatial attention masks
                batch_size = latents.shape[0]
                device = latents.device
                dtype = latents.dtype
                
                # Get sequence lengths
                entity_seq_lens = [mask.sum(dim=1).item() for mask in entity_prompt_emb_mask]
                main_seq_len = prompt_emb_mask.sum(dim=1).item()
                all_seq_lens = entity_seq_lens + [main_seq_len]
                
                # Image sequence length from latents
                img_h, img_w = latents.shape[2], latents.shape[3]
                image_seq_len = img_h * img_w
                
                total_seq_len = sum(all_seq_lens) + image_seq_len
                logger.info(f"FIXED: Sequence lengths - entities: {entity_seq_lens}, main: {main_seq_len}, image: {image_seq_len}, total: {total_seq_len}")
                
                # 3. FIXED: Create proper attention mask for regional control
                attention_mask = torch.zeros((batch_size, total_seq_len, total_seq_len), device=device, dtype=dtype)
                
                # Text sequence starts
                text_start = 0
                text_cumsum = [0]
                for seq_len in all_seq_lens:
                    text_cumsum.append(text_cumsum[-1] + seq_len)
                
                # Image sequence start
                image_start = sum(all_seq_lens)
                
                # FIXED: Apply regional constraints for each entity
                for i in range(len(entity_prompt_emb)):
                    entity_start = text_cumsum[i] 
                    entity_end = text_cumsum[i + 1]
                    
                    # Get entity mask and process to image sequence format
                    entity_mask = entity_masks[0, i, 0]  # [H, W]
                    
                    # Resize entity mask to match latent dimensions
                    entity_mask_resized = F.interpolate(
                        entity_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(img_h, img_w),
                        mode='nearest'
                    ).squeeze().bool()
                    
                    # Flatten to image sequence
                    entity_mask_flat = entity_mask_resized.flatten()  # [H*W]
                    
                    # FIXED: Apply regional attention constraints
                    # Entity prompt can only attend to its masked image regions
                    for seq_idx in range(entity_start, entity_end):
                        for img_idx in range(image_seq_len):
                            if entity_mask_flat[img_idx]:
                                attention_mask[0, seq_idx, image_start + img_idx] = 1.0
                                attention_mask[0, image_start + img_idx, seq_idx] = 1.0
                    
                    logger.debug(f"FIXED: Applied regional constraint for entity {i+1}, mask coverage: {entity_mask_flat.sum().item()}/{image_seq_len}")
                
                # Global prompt can attend to everything
                global_start = text_cumsum[-2]
                global_end = text_cumsum[-1] 
                attention_mask[0, global_start:global_end, image_start:] = 1.0
                attention_mask[0, image_start:, global_start:global_end] = 1.0
                
                # Allow text-to-text attention within sequences
                for i, (start, end) in enumerate(zip(text_cumsum[:-1], text_cumsum[1:])):
                    attention_mask[0, start:end, start:end] = 1.0
                
                # Convert to attention mask format (-inf for masked positions)
                attention_mask = torch.where(attention_mask > 0, torch.zeros_like(attention_mask), torch.full_like(attention_mask, float('-inf')))
                attention_mask = attention_mask.unsqueeze(1)  # Add head dimension
                
                logger.info("FIXED: Created regional attention mask successfully")
                
                return all_prompt_emb, None, attention_mask
                
            except Exception as e:
                logger.error(f"FIXED process_entity_masks failed: {e}")
                import traceback
                traceback.print_exc()
                return prompt_emb, None, None
        
        # Add the FIXED method to model
        model.process_entity_masks = process_entity_masks.__get__(model, model.__class__)
        logger.info("FIXED: Added corrected process_entity_masks method to model")
        return model


class FixedQwenEliGenEntityInput:
    """
    FIXED Entity Input Node - corrected mask preview to show ALL entities
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful girl wearing white dress, holding a mirror, with a forest background",
                    "tooltip": "Global scene description"
                }),
                "entity_prompt_1": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful woman",
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
                    "tooltip": "Second entity description"
                }),
                "entity_mask_2": ("IMAGE", {
                    "tooltip": "Mask for second entity region"
                }),
                "entity_prompt_3": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Third entity description"
                }),
                "entity_mask_3": ("IMAGE", {
                    "tooltip": "Mask for third entity region"
                }),
                "entity_prompt_4": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Fourth entity description"
                }),
                "entity_mask_4": ("IMAGE", {
                    "tooltip": "Mask for fourth entity region"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "ELIGEN_ENTITY_DATA", "IMAGE")
    RETURN_NAMES = ("main_prompt", "entity_data", "mask_preview")
    FUNCTION = "create_entity_data"
    CATEGORY = "conditioning/eligen"

    def create_fixed_mask_preview(self, entity_masks_raw: List, entity_prompts: List, height: int, width: int):
        """
        FIXED mask preview that shows ALL entities, not just entity 1
        """
        try:
            if not entity_masks_raw or not entity_prompts:
                logger.warning("FIXED: No masks or prompts for preview")
                return torch.zeros((1, height, width, 3), dtype=torch.float32)
            
            # Create base canvas
            canvas = Image.new('RGB', (width, height), (0, 0, 0))
            
            # FIXED: Better color palette for multiple entities
            colors = [
                (255, 100, 100, 128),  # Red
                (100, 255, 100, 128),  # Green
                (100, 100, 255, 128),  # Blue
                (255, 255, 100, 128),  # Yellow
                (255, 100, 255, 128),  # Magenta
                (100, 255, 255, 128),  # Cyan
                (255, 165, 100, 128),  # Orange
                (200, 100, 255, 128),  # Purple
            ]
            
            # Load font
            font_size = max(16, int(min(height, width) * 0.025))
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # FIXED: Process ALL entities (was breaking after entity 1)
            logger.info(f"FIXED: Processing {len(entity_masks_raw)} entity masks for preview")
            
            for entity_idx in range(len(entity_masks_raw)):  # FIXED: Explicit range instead of enumerate
                mask_tensor = entity_masks_raw[entity_idx]
                prompt = entity_prompts[entity_idx]
                
                if mask_tensor is None:
                    logger.warning(f"FIXED: Skipping entity {entity_idx+1} - no mask")
                    continue
                
                color = colors[entity_idx % len(colors)]
                
                try:
                    # Convert mask tensor to PIL
                    if isinstance(mask_tensor, torch.Tensor):
                        if len(mask_tensor.shape) == 4:
                            mask_array = mask_tensor[0].cpu().numpy()
                        else:
                            mask_array = mask_tensor.cpu().numpy()
                        
                        # Handle channels
                        if len(mask_array.shape) == 3:
                            if mask_array.shape[2] == 3:
                                mask_array = mask_array.mean(axis=2)
                            elif mask_array.shape[2] == 1:
                                mask_array = mask_array[:, :, 0]
                        
                        # Normalize to 0-255
                        if mask_array.max() <= 1.0:
                            mask_array = (mask_array * 255).astype(np.uint8)
                        
                        # Create PIL mask
                        mask_pil = Image.fromarray(mask_array, mode='L')
                        if mask_pil.size != (width, height):
                            mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                    else:
                        mask_pil = mask_tensor.convert('L').resize((width, height), Image.NEAREST)
                    
                    # FIXED: Create colored overlay for this entity
                    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                    overlay_pixels = overlay.load()
                    mask_pixels = mask_pil.load()
                    
                    # Apply color to mask regions
                    for y in range(height):
                        for x in range(width):
                            if mask_pixels[x, y] > 127:  # White area
                                overlay_pixels[x, y] = color
                    
                    # Add text label
                    if font is not None:
                        draw = ImageDraw.Draw(overlay)
                        mask_bbox = mask_pil.getbbox()
                        
                        if mask_bbox:
                            x0, y0, x1, y1 = mask_bbox
                            text_x = max(10, x0 + 10)
                            text_y = max(10, y0 + 10)
                            
                            # Background for text
                            try:
                                text_bbox = draw.textbbox((text_x, text_y), prompt, font=font)
                                bg_x0, bg_y0, bg_x1, bg_y1 = text_bbox
                                draw.rectangle([bg_x0-2, bg_y0-1, bg_x1+2, bg_y1+1], fill=(0, 0, 0, 200))
                            except:
                                pass
                            
                            # Draw text
                            draw.text((text_x, text_y), prompt, fill=(255, 255, 255, 255), font=font)
                    
                    # Composite onto canvas
                    canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
                    logger.info(f"FIXED: Added entity {entity_idx+1} '{prompt}' to preview")
                    
                except Exception as e:
                    logger.error(f"FIXED: Failed to process entity {entity_idx+1}: {e}")
                    continue
            
            # Convert to tensor
            result_array = np.array(canvas).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_array).unsqueeze(0)
            
            logger.info(f"FIXED: Created preview with ALL {len(entity_prompts)} entities")
            return result_tensor
            
        except Exception as e:
            logger.error(f"FIXED: Mask preview creation failed: {e}")
            # Fallback preview
            fallback = Image.new('RGB', (width, height), (50, 50, 50))
            draw = ImageDraw.Draw(fallback)
            
            # Draw colored rectangles for each entity
            for i in range(min(len(entity_prompts), 4)):
                color = colors[i][:3]  # Remove alpha
                x = (width // 5) * (i + 1)
                y = height // 2
                size = 40
                draw.rectangle([x-size, y-size, x+size, y+size], fill=color)
                if font:
                    draw.text((x-size, y+size+5), f"E{i+1}", fill=(255, 255, 255), font=font)
            
            fallback_array = np.array(fallback).astype(np.float32) / 255.0
            return torch.from_numpy(fallback_array).unsqueeze(0)

    def preprocess_masks_for_diffsynth(self, masks: List, height: int, width: int):
        """Preprocess masks for DiffSynth format"""
        out_masks = []
        latent_height, latent_width = height // 8, width // 8
        
        for i, mask_tensor in enumerate(masks):
            try:
                # Convert to PIL
                if isinstance(mask_tensor, torch.Tensor):
                    if len(mask_tensor.shape) == 4:
                        mask_array = mask_tensor[0].cpu().numpy()
                    else:
                        mask_array = mask_tensor.cpu().numpy()
                    
                    if len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
                        mask_array = mask_array.mean(axis=2)
                    elif len(mask_array.shape) == 3 and mask_array.shape[2] == 1:
                        mask_array = mask_array[:, :, 0]
                    
                    if mask_array.max() <= 1.0:
                        mask_array = (mask_array * 255).astype(np.uint8)
                    
                    mask_pil = Image.fromarray(mask_array.astype(np.uint8), mode='L')
                else:
                    mask_pil = mask_tensor.convert('L')
                
                # Resize to latent resolution
                mask_pil = mask_pil.resize((latent_width, latent_height), resample=Image.NEAREST)
                
                # Convert to binary tensor
                mask_array = np.array(mask_pil).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                mask_tensor = (mask_tensor > 0.5).float()
                
                out_masks.append(mask_tensor)
                logger.debug(f"FIXED: Preprocessed mask {i+1}: {mask_tensor.shape}")
                
            except Exception as e:
                logger.error(f"FIXED: Failed to preprocess mask {i+1}: {e}")
                # Dummy mask
                dummy_mask = torch.zeros(1, 1, latent_height, latent_width, dtype=torch.float32)
                out_masks.append(dummy_mask)
        
        return out_masks

    def create_entity_data(self, global_prompt: str, entity_prompt_1: str, entity_mask_1,
                          entity_prompt_2: str = "", entity_mask_2=None,
                          entity_prompt_3: str = "", entity_mask_3=None,
                          entity_prompt_4: str = "", entity_mask_4=None,
                          height: int = 1024, width: int = 1024):
        """Create entity data with FIXED preview and preprocessing"""
        try:
            # Collect entities
            entity_prompts = []
            entity_masks_raw = []
            
            entities = [
                (entity_prompt_1, entity_mask_1),
                (entity_prompt_2, entity_mask_2),
                (entity_prompt_3, entity_mask_3),
                (entity_prompt_4, entity_mask_4),
            ]
            
            for prompt, mask in entities:
                if prompt and prompt.strip() and mask is not None:
                    entity_prompts.append(prompt.strip())
                    entity_masks_raw.append(mask)
            
            if not entity_prompts:
                logger.warning("FIXED: No valid entities provided")
                empty_preview = torch.zeros((1, height, width, 3), dtype=torch.float32)
                return (global_prompt, {"entity_prompts": [], "entity_masks": [], "num_entities": 0}, empty_preview)
            
            # FIXED: Create preview showing ALL entities
            mask_preview = self.create_fixed_mask_preview(entity_masks_raw, entity_prompts, height, width)
            
            # Preprocess masks
            entity_masks = self.preprocess_masks_for_diffsynth(entity_masks_raw, height, width)
            
            # Create entity data
            entity_data = {
                "entity_prompts": entity_prompts,
                "entity_masks": entity_masks,
                "height": height,
                "width": width,
                "num_entities": len(entity_prompts)
            }
            
            logger.info(f"FIXED: Created {len(entity_prompts)} entities with corrected preview")
            return (global_prompt, entity_data, mask_preview)
            
        except Exception as e:
            logger.error(f"FIXED: Entity data creation failed: {e}")
            empty_preview = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (global_prompt, {"entity_prompts": [], "entity_masks": [], "num_entities": 0}, empty_preview)


class FixedQwenEliGenApply:
    """
    FIXED Apply Node - corrected regional entity control
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "entity_data": ("ELIGEN_ENTITY_DATA",),
                "clip": ("CLIP",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_eligen"
    CATEGORY = "conditioning/eligen"

    def encode_entity_prompt(self, clip, prompt: str):
        """Encode entity prompt with DiffSynth template"""
        template = "<|im_start|>system\\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\\n<|im_start|>user\\n{}\\n<|im_end|>\\n<|im_start|>assistant\\n"
        formatted_prompt = template.format(prompt)
        
        tokens = clip.tokenize(formatted_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        return {"prompt_emb": cond, "pooled": pooled}

    def apply_eligen(self, model, entity_data: Dict, clip, strength: float = 1.0):
        """FIXED entity control application with proper regional constraints"""
        try:
            if not entity_data or not entity_data.get("entity_prompts") or entity_data.get("num_entities", 0) == 0:
                logger.warning("FIXED: No entity data provided")
                return (model,)
            
            entity_prompts = entity_data["entity_prompts"]
            entity_masks = entity_data["entity_masks"]
            
            logger.info(f"FIXED: Applying regional entity control for {len(entity_prompts)} entities")
            
            # Clone and patch model
            patched_model = model.clone()
            
            if hasattr(patched_model.model, 'diffusion_model'):
                diffusion_model = patched_model.model.diffusion_model
                FixedProcessEntityMasksExtension.add_process_entity_masks_to_model(diffusion_model)
                logger.info("FIXED: Added corrected process_entity_masks method")
            else:
                logger.error("FIXED: No diffusion_model found")
                return (model,)
            
            def patched_apply_model(original_apply_model):
                def apply_model_wrapper(x, timestep, context=None, **kwargs):
                    try:
                        if context is None:
                            return original_apply_model(x, timestep, context, **kwargs)
                        
                        # Encode entity prompts
                        entity_prompt_embeds = []
                        entity_prompt_masks = []
                        
                        for i, prompt in enumerate(entity_prompts):
                            try:
                                encoded = self.encode_entity_prompt(clip, prompt)
                                entity_prompt_embeds.append(encoded["prompt_emb"])
                                seq_len = encoded["prompt_emb"].shape[1]
                                attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=encoded["prompt_emb"].device)
                                entity_prompt_masks.append(attention_mask)
                                logger.debug(f"FIXED: Encoded entity {i+1}: '{prompt}'")
                            except Exception as e:
                                logger.error(f"FIXED: Failed to encode entity {i+1}: {e}")
                                continue
                        
                        if not entity_prompt_embeds:
                            logger.warning("FIXED: No entity prompts encoded")
                            return original_apply_model(x, timestep, context, **kwargs)
                        
                        # Process masks
                        processed_entity_masks = []
                        for i, mask_tensor in enumerate(entity_masks):
                            try:
                                mask = mask_tensor.to(device=x.device, dtype=x.dtype)
                                processed_entity_masks.append(mask)
                            except Exception as e:
                                logger.error(f"FIXED: Failed to process mask {i+1}: {e}")
                                continue
                        
                        if processed_entity_masks:
                            entity_masks_tensor = torch.cat(processed_entity_masks, dim=0).unsqueeze(0)
                        else:
                            return original_apply_model(x, timestep, context, **kwargs)
                        
                        # Get dimensions
                        height, width = x.shape[2] * 16, x.shape[3] * 16
                        
                        # FIXED: Use regional process_entity_masks
                        diffusion_model = patched_model.model.diffusion_model
                        if hasattr(diffusion_model, 'process_entity_masks'):
                            try:
                                # Create prompt mask
                                prompt_emb_mask = torch.ones(context.shape[0], context.shape[1], 
                                                           dtype=torch.long, device=context.device)
                                
                                # Process image
                                image = rearrange(x, "B C (H P) (W Q) -> B (H W) (C P Q)", 
                                                H=height//16, W=width//16, P=2, Q=2)
                                if hasattr(diffusion_model, 'img_in'):
                                    image = diffusion_model.img_in(image)
                                
                                img_shapes = [(x.shape[0], x.shape[2]*2, x.shape[3]*2)]
                                
                                # FIXED: Call corrected process_entity_masks
                                text, image_rotary_emb, attention_mask = diffusion_model.process_entity_masks(
                                    x, context, prompt_emb_mask, entity_prompt_embeds, entity_prompt_masks,
                                    entity_masks_tensor, height, width, image, img_shapes
                                )
                                
                                # FIXED: Apply regional attention mask
                                if attention_mask is not None:
                                    kwargs['attention_mask'] = attention_mask
                                    logger.info("FIXED: Applied regional attention mask")
                                
                                if text is not None:
                                    context = text
                                    logger.info("FIXED: Applied entity-enhanced context")
                                
                            except Exception as e:
                                logger.error(f"FIXED: process_entity_masks failed: {e}")
                        
                        return original_apply_model(x, timestep, context, **kwargs)
                        
                    except Exception as e:
                        logger.error(f"FIXED: Wrapper failed: {e}")
                        return original_apply_model(x, timestep, context, **kwargs)
                
                return apply_model_wrapper
            
            # Apply patch
            original_apply_model = patched_model.model.apply_model
            patched_model.model.apply_model = patched_apply_model(original_apply_model)
            
            logger.info(f"FIXED: Regional entity control applied for {len(entity_prompts)} entities")
            return (patched_model,)
            
        except Exception as e:
            logger.error(f"FIXED: Application failed: {e}")
            return (model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FixedQwenEliGenEntityInput": FixedQwenEliGenEntityInput,
    "FixedQwenEliGenApply": FixedQwenEliGenApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FixedQwenEliGenEntityInput": "🎭 Qwen EliGen Entity Input (FIXED)",
    "FixedQwenEliGenApply": "🎯 Qwen EliGen Apply (FIXED)",
}

__version__ = "2.1.0"
__description__ = "FIXED EliGen implementation - corrects mask preview and regional entity control"
