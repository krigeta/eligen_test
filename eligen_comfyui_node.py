"""
DEEP FIX QWEN ELIGEN CUSTOM NODE FOR COMFYUI
Fixes the REAL issue: ComfyUI's attention mechanism doesn't use attention_mask
Patches the actual attention layers directly for regional entity control
Version: 2.2.0 (Deep Architecture Fix)
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


class DeepFixProcessEntityMasksExtension:
    """
    DEEP FIX: Patches ComfyUI's actual attention mechanism for regional entity control
    This addresses the core issue that ComfyUI doesn't use attention_mask in transformer blocks
    """
    
    @staticmethod
    def patch_attention_mechanism(model):
        """
        DEEP FIX: Patch the actual attention mechanism in QwenImageTransformer2DModel
        This makes ComfyUI's attention layers aware of entity masks
        """
        
        def create_entity_aware_attention_patch():
            """Create a patch that modifies attention to respect entity regions"""
            
            def attention_patch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, entity_attention_mask=None, **kwargs):
                """
                DEEP FIX: Modified attention that respects entity regional constraints
                This is the actual fix - modifying the attention computation directly
                """
                try:
                    # If we have entity attention mask, apply it
                    if entity_attention_mask is not None:
                        logger.info(f"DEEP FIX: Applying entity attention mask in attention layer")
                        
                        # Get dimensions
                        b, seq_len, dim = q.shape
                        head_dim = dim // heads
                        
                        # Reshape for multi-head attention
                        q = q.view(b, seq_len, heads, head_dim).transpose(1, 2)  # [B, H, S, D]
                        k = k.view(b, seq_len, heads, head_dim).transpose(1, 2)
                        v = v.view(b, seq_len, heads, head_dim).transpose(1, 2)
                        
                        # Compute attention scores
                        scale = head_dim ** -0.5
                        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]
                        
                        # DEEP FIX: Apply entity attention mask
                        if entity_attention_mask.shape[1] == 1:  # Single head mask
                            entity_mask_expanded = entity_attention_mask.expand(-1, heads, -1, -1)
                        else:
                            entity_mask_expanded = entity_attention_mask
                        
                        # Ensure mask dimensions match attention scores
                        if entity_mask_expanded.shape[2:] == attn_scores.shape[2:]:
                            attn_scores = attn_scores + entity_mask_expanded
                            logger.debug(f"DEEP FIX: Applied entity mask to attention scores")
                        else:
                            logger.warning(f"DEEP FIX: Entity mask dimension mismatch: {entity_mask_expanded.shape} vs {attn_scores.shape}")
                        
                        # Apply softmax
                        attn_probs = F.softmax(attn_scores, dim=-1)
                        
                        # Apply to values
                        out = torch.matmul(attn_probs, v)  # [B, H, S, D]
                        
                        # Reshape back
                        out = out.transpose(1, 2).contiguous().view(b, seq_len, dim)
                        
                        logger.debug(f"DEEP FIX: Entity-aware attention computed successfully")
                        return out
                    
                    # Fallback to original attention if no entity mask
                    logger.debug("DEEP FIX: No entity mask, using standard attention")
                    
                    # Use ComfyUI's optimized attention as fallback
                    from comfy.ldm.modules.attention import optimized_attention_masked
                    return optimized_attention_masked(q, k, v, heads, mask, **kwargs)
                    
                except Exception as e:
                    logger.error(f"DEEP FIX: Entity-aware attention failed: {e}")
                    # Always fallback to standard attention
                    from comfy.ldm.modules.attention import optimized_attention_masked
                    return optimized_attention_masked(q, k, v, heads, mask, **kwargs)
            
            return attention_patch
        
        # DEEP FIX: Replace ComfyUI's attention function with entity-aware version
        try:
            import comfy.ldm.modules.attention
            original_attention = comfy.ldm.modules.attention.optimized_attention_masked
            
            # Create entity-aware attention wrapper
            entity_aware_attention = create_entity_aware_attention_patch()
            
            # Store original for restoration
            if not hasattr(comfy.ldm.modules.attention, '_original_optimized_attention_masked'):
                comfy.ldm.modules.attention._original_optimized_attention_masked = original_attention
            
            # Replace with entity-aware version
            comfy.ldm.modules.attention.optimized_attention_masked = entity_aware_attention
            
            logger.info("DEEP FIX: Successfully patched ComfyUI's attention mechanism")
            return True
            
        except Exception as e:
            logger.error(f"DEEP FIX: Failed to patch attention mechanism: {e}")
            return False
    
    @staticmethod
    def add_process_entity_masks_to_model(model):
        """
        DEEP FIX: Enhanced process_entity_masks with proper regional attention
        """
        
        def process_entity_masks(self, latents, prompt_emb, prompt_emb_mask, 
                                entity_prompt_emb, entity_prompt_emb_mask, 
                                entity_masks, height, width, image, img_shapes):
            """
            DEEP FIX: Enhanced process_entity_masks that creates proper regional constraints
            """
            try:
                logger.info(f"DEEP FIX: Processing {len(entity_prompt_emb)} entities for DEEP regional control")
                
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
                
                all_prompt_emb = torch.cat(processed_embeddings, dim=1)
                logger.info(f"DEEP FIX: Combined embeddings shape: {all_prompt_emb.shape}")
                
                # 2. DEEP FIX: Create enhanced spatial attention masks
                batch_size = latents.shape[0]
                device = latents.device
                dtype = latents.dtype
                
                # Get sequence lengths
                entity_seq_lens = [mask.sum(dim=1).item() for mask in entity_prompt_emb_mask]
                main_seq_len = prompt_emb_mask.sum(dim=1).item()
                all_seq_lens = entity_seq_lens + [main_seq_len]
                
                # Image sequence length from latents
                img_h, img_w = latents.shape[2], latents.shape[3]  # Latent dimensions
                image_seq_len = img_h * img_w
                
                total_seq_len = sum(all_seq_lens) + image_seq_len
                logger.info(f"DEEP FIX: Enhanced sequence setup - entities: {entity_seq_lens}, main: {main_seq_len}, image: {image_seq_len}")
                
                # 3. DEEP FIX: Create ENHANCED attention mask for regional control
                attention_mask = torch.full((batch_size, total_seq_len, total_seq_len), 
                                          float('-inf'), device=device, dtype=dtype)
                
                # Text sequence positions
                text_cumsum = [0]
                for seq_len in all_seq_lens:
                    text_cumsum.append(text_cumsum[-1] + seq_len)
                
                # Image sequence start
                image_start = sum(all_seq_lens)
                
                # DEEP FIX: Apply ENHANCED regional constraints for each entity
                for i in range(len(entity_prompt_emb)):
                    entity_start = text_cumsum[i] 
                    entity_end = text_cumsum[i + 1]
                    
                    # Get entity mask - DEEP FIX: Proper mask processing
                    entity_mask = entity_masks[0, i, 0]  # [H_mask, W_mask]
                    
                    # DEEP FIX: Resize to exact latent dimensions
                    entity_mask_resized = F.interpolate(
                        entity_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(img_h, img_w),
                        mode='nearest'
                    ).squeeze().bool()
                    
                    # Flatten to image sequence
                    entity_mask_flat = entity_mask_resized.flatten()  # [H*W]
                    
                    # DEEP FIX: Apply STRICT regional attention constraints
                    # Entity can only attend to itself and its masked image regions
                    
                    # 1. Entity self-attention (within entity prompt)
                    attention_mask[0, entity_start:entity_end, entity_start:entity_end] = 0.0
                    
                    # 2. Entity -> masked image regions ONLY
                    for seq_idx in range(entity_start, entity_end):
                        for img_idx in range(image_seq_len):
                            if entity_mask_flat[img_idx]:
                                attention_mask[0, seq_idx, image_start + img_idx] = 0.0
                                attention_mask[0, image_start + img_idx, seq_idx] = 0.0
                    
                    # 3. DEEP FIX: Block cross-entity attention (entities can't see each other)
                    for j in range(len(entity_prompt_emb)):
                        if i != j:
                            other_start = text_cumsum[j]
                            other_end = text_cumsum[j + 1]
                            # Block entity i from attending to entity j
                            attention_mask[0, entity_start:entity_end, other_start:other_end] = float('-inf')
                    
                    mask_coverage = entity_mask_flat.sum().item()
                    logger.info(f"DEEP FIX: Entity {i+1} regional constraint applied - coverage: {mask_coverage}/{image_seq_len} pixels")
                
                # DEEP FIX: Global prompt can attend to everything
                global_start = text_cumsum[-2]
                global_end = text_cumsum[-1]
                
                # Global self-attention
                attention_mask[0, global_start:global_end, global_start:global_end] = 0.0
                
                # Global can attend to all image regions
                attention_mask[0, global_start:global_end, image_start:] = 0.0
                attention_mask[0, image_start:, global_start:global_end] = 0.0
                
                # Global can attend to all entities (for composition)
                for i in range(len(entity_prompt_emb)):
                    entity_start = text_cumsum[i]
                    entity_end = text_cumsum[i + 1]
                    attention_mask[0, global_start:global_end, entity_start:entity_end] = 0.0
                    attention_mask[0, entity_start:entity_end, global_start:global_end] = 0.0
                
                # Add head dimension for multi-head attention
                attention_mask = attention_mask.unsqueeze(1)  # [B, 1, S, S]
                
                logger.info("DEEP FIX: Enhanced regional attention mask created successfully")
                
                return all_prompt_emb, None, attention_mask
                
            except Exception as e:
                logger.error(f"DEEP FIX process_entity_masks failed: {e}")
                import traceback
                traceback.print_exc()
                return prompt_emb, None, None
        
        # Add the DEEP FIX method to model
        model.process_entity_masks = process_entity_masks.__get__(model, model.__class__)
        logger.info("DEEP FIX: Added enhanced process_entity_masks method to model")
        return model


class DeepFixQwenEliGenEntityInput:
    """
    DEEP FIX Entity Input Node - Enhanced for better regional control testing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful scene with multiple subjects",
                    "tooltip": "Global scene description (background, overall composition)"
                }),
                "entity_prompt_1": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful woman with long hair",
                    "tooltip": "First entity description - what appears in mask 1"
                }),
                "entity_mask_1": ("IMAGE", {
                    "tooltip": "Mask for first entity region (white=entity, black=background)"
                }),
                "entity_prompt_2": ("STRING", {
                    "multiline": True,
                    "default": "A golden mirror reflecting light",
                    "tooltip": "Second entity description - what appears in mask 2"
                }),
                "entity_mask_2": ("IMAGE", {
                    "tooltip": "Mask for second entity region (white=entity, black=background)"
                }),
            },
            "optional": {
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

    def create_enhanced_mask_preview(self, entity_masks_raw: List, entity_prompts: List, height: int, width: int):
        """
        DEEP FIX: Enhanced mask preview that clearly shows ALL entities
        """
        try:
            if not entity_masks_raw or not entity_prompts:
                logger.warning("DEEP FIX: No masks or prompts for preview")
                return torch.zeros((1, height, width, 3), dtype=torch.float32)
            
            # Create high contrast canvas
            canvas = Image.new('RGB', (width, height), (20, 20, 20))
            
            # DEEP FIX: High contrast colors for clear distinction
            colors = [
                (255, 80, 80),    # Bright Red
                (80, 255, 80),    # Bright Green  
                (80, 80, 255),    # Bright Blue
                (255, 255, 80),   # Bright Yellow
                (255, 80, 255),   # Bright Magenta
                (80, 255, 255),   # Bright Cyan
                (255, 165, 80),   # Bright Orange
                (200, 80, 255),   # Bright Purple
            ]
            
            # Load font
            font_size = max(20, int(min(height, width) * 0.03))
            font = None
            try:
                font = ImageFont.load_default()
            except:
                pass
            
            # DEEP FIX: Process ALL entities with clear separation
            logger.info(f"DEEP FIX: Creating enhanced preview for {len(entity_masks_raw)} entities")
            
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            
            for entity_idx in range(len(entity_masks_raw)):
                mask_tensor = entity_masks_raw[entity_idx]
                prompt = entity_prompts[entity_idx]
                
                if mask_tensor is None or not prompt.strip():
                    logger.warning(f"DEEP FIX: Skipping entity {entity_idx+1} - missing mask or prompt")
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
                        
                        # Normalize
                        if mask_array.max() <= 1.0:
                            mask_array = (mask_array * 255).astype(np.uint8)
                        
                        mask_pil = Image.fromarray(mask_array, mode='L')
                        if mask_pil.size != (width, height):
                            mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                    else:
                        mask_pil = mask_tensor.convert('L').resize((width, height), Image.NEAREST)
                    
                    # DEEP FIX: Create high-contrast overlay
                    entity_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                    entity_pixels = entity_overlay.load()
                    mask_pixels = mask_pil.load()
                    
                    # Apply bright color to mask regions
                    for y in range(height):
                        for x in range(width):
                            if mask_pixels[x, y] > 127:  # White area
                                entity_pixels[x, y] = (*color, 160)  # Semi-transparent
                    
                    # Add clear text label
                    if font is not None:
                        draw = ImageDraw.Draw(entity_overlay)
                        mask_bbox = mask_pil.getbbox()
                        
                        if mask_bbox:
                            x0, y0, x1, y1 = mask_bbox
                            text_x = max(15, x0 + 15)
                            text_y = max(15, y0 + 15)
                            
                            # High contrast background for text
                            label = f"E{entity_idx+1}: {prompt[:30]}..."
                            try:
                                text_bbox = draw.textbbox((text_x, text_y), label, font=font)
                                bg_x0, bg_y0, bg_x1, bg_y1 = text_bbox
                                draw.rectangle([bg_x0-3, bg_y0-2, bg_x1+3, bg_y1+2], fill=(0, 0, 0, 220))
                            except:
                                pass
                            
                            # Draw bright text
                            draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)
                    
                    # Composite this entity
                    overlay = Image.alpha_composite(overlay, entity_overlay)
                    logger.info(f"DEEP FIX: Added entity {entity_idx+1} '{prompt[:20]}...' to preview")
                    
                except Exception as e:
                    logger.error(f"DEEP FIX: Failed to process entity {entity_idx+1}: {e}")
                    continue
            
            # Final composition
            canvas_rgba = canvas.convert('RGBA')
            result = Image.alpha_composite(canvas_rgba, overlay).convert('RGB')
            
            # Convert to tensor
            result_array = np.array(result).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_array).unsqueeze(0)
            
            logger.info(f"DEEP FIX: Enhanced preview created with {len(entity_prompts)} entities")
            return result_tensor
            
        except Exception as e:
            logger.error(f"DEEP FIX: Enhanced preview creation failed: {e}")
            # Simple fallback
            fallback = Image.new('RGB', (width, height), (60, 60, 60))
            draw = ImageDraw.Draw(fallback)
            
            for i in range(min(len(entity_prompts), 4)):
                color = colors[i] if i < len(colors) else (200, 200, 200)
                x = (width // 6) * (i + 1)
                y = height // 2
                size = 50
                draw.rectangle([x-size, y-size, x+size, y+size], fill=color)
                draw.text((x-size, y+size+10), f"E{i+1}", fill=(255, 255, 255))
            
            fallback_array = np.array(fallback).astype(np.float32) / 255.0
            return torch.from_numpy(fallback_array).unsqueeze(0)

    def preprocess_masks_for_diffsynth(self, masks: List, height: int, width: int):
        """DEEP FIX: Enhanced mask preprocessing"""
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
                
                # DEEP FIX: Resize to exact latent resolution
                mask_pil = mask_pil.resize((latent_width, latent_height), resample=Image.NEAREST)
                
                # Convert to binary tensor
                mask_array = np.array(mask_pil).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                mask_tensor = (mask_tensor > 0.5).float()
                
                out_masks.append(mask_tensor)
                logger.debug(f"DEEP FIX: Preprocessed mask {i+1}: {mask_tensor.shape}, coverage: {mask_tensor.sum().item()}")
                
            except Exception as e:
                logger.error(f"DEEP FIX: Failed to preprocess mask {i+1}: {e}")
                dummy_mask = torch.zeros(1, 1, latent_height, latent_width, dtype=torch.float32)
                out_masks.append(dummy_mask)
        
        return out_masks

    def create_entity_data(self, global_prompt: str, entity_prompt_1: str, entity_mask_1,
                          entity_prompt_2: str = "", entity_mask_2=None,
                          entity_prompt_3: str = "", entity_mask_3=None,
                          entity_prompt_4: str = "", entity_mask_4=None,
                          height: int = 1024, width: int = 1024):
        """DEEP FIX: Enhanced entity data creation"""
        try:
            # DEEP FIX: Collect ALL entities (including required entity 2)
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
            
            if len(entity_prompts) < 1:
                logger.warning("DEEP FIX: Need at least 1 entity")
                empty_preview = torch.zeros((1, height, width, 3), dtype=torch.float32)
                return (global_prompt, {"entity_prompts": [], "entity_masks": [], "num_entities": 0}, empty_preview)
            
            # DEEP FIX: Enhanced preview showing ALL entities clearly
            mask_preview = self.create_enhanced_mask_preview(entity_masks_raw, entity_prompts, height, width)
            
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
            
            logger.info(f"DEEP FIX: Created {len(entity_prompts)} entities with enhanced preview")
            logger.info(f"DEEP FIX: Entity prompts: {entity_prompts}")
            
            return (global_prompt, entity_data, mask_preview)
            
        except Exception as e:
            logger.error(f"DEEP FIX: Entity data creation failed: {e}")
            empty_preview = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (global_prompt, {"entity_prompts": [], "entity_masks": [], "num_entities": 0}, empty_preview)


class DeepFixQwenEliGenApply:
    """
    DEEP FIX Apply Node - Patches the actual attention mechanism
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
        """DEEP FIX: Apply entity control by patching the actual attention mechanism"""
        try:
            if not entity_data or not entity_data.get("entity_prompts") or entity_data.get("num_entities", 0) == 0:
                logger.warning("DEEP FIX: No entity data provided")
                return (model,)
            
            entity_prompts = entity_data["entity_prompts"]
            entity_masks = entity_data["entity_masks"]
            
            logger.info(f"DEEP FIX: Applying DEEP entity control for {len(entity_prompts)} entities")
            
            # Clone model
            patched_model = model.clone()
            
            # DEEP FIX: Patch the actual attention mechanism
            success = DeepFixProcessEntityMasksExtension.patch_attention_mechanism(patched_model)
            if not success:
                logger.error("DEEP FIX: Failed to patch attention mechanism")
                return (model,)
            
            # Add enhanced process_entity_masks method
            if hasattr(patched_model.model, 'diffusion_model'):
                diffusion_model = patched_model.model.diffusion_model
                DeepFixProcessEntityMasksExtension.add_process_entity_masks_to_model(diffusion_model)
                logger.info("DEEP FIX: Added enhanced process_entity_masks method")
            else:
                logger.error("DEEP FIX: No diffusion_model found")
                return (model,)
            
            def deep_patched_apply_model(original_apply_model):
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
                                logger.debug(f"DEEP FIX: Encoded entity {i+1}: '{prompt[:30]}...'")
                            except Exception as e:
                                logger.error(f"DEEP FIX: Failed to encode entity {i+1}: {e}")
                                continue
                        
                        if not entity_prompt_embeds:
                            return original_apply_model(x, timestep, context, **kwargs)
                        
                        # Process masks
                        processed_entity_masks = []
                        for i, mask_tensor in enumerate(entity_masks):
                            try:
                                mask = mask_tensor.to(device=x.device, dtype=x.dtype)
                                processed_entity_masks.append(mask)
                            except Exception as e:
                                logger.error(f"DEEP FIX: Failed to process mask {i+1}: {e}")
                                continue
                        
                        if processed_entity_masks:
                            entity_masks_tensor = torch.cat(processed_entity_masks, dim=0).unsqueeze(0)
                        else:
                            return original_apply_model(x, timestep, context, **kwargs)
                        
                        # Get dimensions
                        height, width = x.shape[2] * 16, x.shape[3] * 16
                        
                        # DEEP FIX: Use enhanced process_entity_masks
                        diffusion_model = patched_model.model.diffusion_model
                        if hasattr(diffusion_model, 'process_entity_masks'):
                            try:
                                prompt_emb_mask = torch.ones(context.shape[0], context.shape[1], 
                                                           dtype=torch.long, device=context.device)
                                
                                # Process image
                                image = rearrange(x, "B C (H P) (W Q) -> B (H W) (C P Q)", 
                                                H=height//16, W=width//16, P=2, Q=2)
                                if hasattr(diffusion_model, 'img_in'):
                                    image = diffusion_model.img_in(image)
                                
                                img_shapes = [(x.shape[0], x.shape[2]*2, x.shape[3]*2)]
                                
                                # DEEP FIX: Call enhanced process_entity_masks
                                text, image_rotary_emb, attention_mask = diffusion_model.process_entity_masks(
                                    x, context, prompt_emb_mask, entity_prompt_embeds, entity_prompt_masks,
                                    entity_masks_tensor, height, width, image, img_shapes
                                )
                                
                                # DEEP FIX: Pass entity attention mask to the attention mechanism
                                if attention_mask is not None:
                                    kwargs['entity_attention_mask'] = attention_mask
                                    logger.info("DEEP FIX: Passed entity attention mask to attention layers")
                                
                                if text is not None:
                                    context = text
                                    logger.info("DEEP FIX: Applied entity-enhanced context")
                                
                            except Exception as e:
                                logger.error(f"DEEP FIX: Enhanced process_entity_masks failed: {e}")
                        
                        return original_apply_model(x, timestep, context, **kwargs)
                        
                    except Exception as e:
                        logger.error(f"DEEP FIX: Wrapper failed: {e}")
                        return original_apply_model(x, timestep, context, **kwargs)
                
                return apply_model_wrapper
            
            # Apply the deep patch
            original_apply_model = patched_model.model.apply_model
            patched_model.model.apply_model = deep_patched_apply_model(original_apply_model)
            
            logger.info(f"DEEP FIX: Successfully applied deep entity control for {len(entity_prompts)} entities")
            return (patched_model,)
            
        except Exception as e:
            logger.error(f"DEEP FIX: Application completely failed: {e}")
            return (model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "DeepFixQwenEliGenEntityInput": DeepFixQwenEliGenEntityInput,
    "DeepFixQwenEliGenApply": DeepFixQwenEliGenApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepFixQwenEliGenEntityInput": "🔧 Qwen EliGen Entity Input (DEEP FIX)",
    "DeepFixQwenEliGenApply": "⚡ Qwen EliGen Apply (DEEP FIX)",
}

__version__ = "2.2.0"
__description__ = "DEEP FIX EliGen - patches actual ComfyUI attention mechanism for regional entity control"

# DEEP FIX USAGE NOTES
DEEP_FIX_NOTES = """
DEEP FIX QWEN ELIGEN - USAGE INSTRUCTIONS

🎯 THE DEEP FIX SOLUTION:
This version patches ComfyUI's actual attention mechanism instead of just the model wrapper.
It directly modifies how attention is computed to respect entity regions.

📋 SETUP FOR TESTING:
1. Replace nodes with: DeepFixQwenEliGenEntityInput + DeepFixQwenEliGenApply
2. CRITICAL: Add at least 2 entities to test regional control:
   - entity_prompt_1: "A beautiful woman with long hair" + mask
   - entity_prompt_2: "A golden mirror reflecting light" + mask
3. Make sure masks are distinct regions (don't overlap)
4. Global prompt: "A beautiful scene with multiple subjects"

✅ EXPECTED RESULTS:
- Entity 1 prompt only affects mask 1 region
- Entity 2 prompt only affects mask 2 region  
- Global prompt affects background/composition
- NO universal prompt behavior

⚠️  IMPORTANT:
This patches ComfyUI's core attention mechanism. If it causes issues,
restart ComfyUI to restore original attention.

🔍 CONSOLE OUTPUT TO LOOK FOR:
- "DEEP FIX: Successfully patched ComfyUI's attention mechanism"
- "DEEP FIX: Processing X entities for DEEP regional control"
- "DEEP FIX: Applying entity attention mask in attention layer"
"""

if __name__ == "__main__":
    print(DEEP_FIX_NOTES)
