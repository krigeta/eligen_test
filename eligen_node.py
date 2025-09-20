import torch
import torch.nn.functional as F
import numpy as np
import comfy.model_patcher
import comfy.utils
import gc
from functools import partial

# A global variable to hold the original attention function
original_optimized_attention = None

class EliGenRegionalControl:
    """
    The main node for applying EliGen's regional attention mechanism.
    It patches the model's attention function and augments conditioning
    to allow for fine-grained, entity-level control over the image generation process.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "global_positive": ("CONDITIONING",),
                "global_negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "eligen_lora": ("STRING", {"default": "FLUX.1-dev-EliGen.safetensors"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
                "region_1_prompt": ("CONDITIONING",),
                "region_1_mask": ("MASK",),
                "region_2_prompt": ("CONDITIONING",),
                "region_2_mask": ("MASK",),
                "region_3_prompt": ("CONDITIONING",),
                "region_3_mask": ("MASK",),
                "region_4_prompt": ("CONDITIONING",),
                "region_4_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "latent")
    FUNCTION = "apply_regional_control"
    CATEGORY = "conditioning/eligen"

    def apply_regional_control(self, model, global_positive, global_negative, latent, eligen_lora, lora_strength, enabled=True, **kwargs):
        """
        Orchestrates the model patching and conditioning augmentation process.
        """
        if not enabled:
            # If disabled, pass through the original inputs without modification.
            return (model, global_positive, global_negative, latent)

        # 1. Clone the model to prevent patch leakage [4]
        patched_model = model.clone()

        # 2. Apply the specific EliGen LoRA
        # This requires a separate loader, but for simplicity, we assume it's loaded via standard nodes.
        # The user should use a "Load LoRA" node before this node.
        # For robustness, we can try to load it here if a utility is available.
        # Note: ComfyUI's core doesn't provide a direct `load_lora_by_name` function inside a node.
        # The standard workflow is to use a `LoraLoader` node. This node will focus on the attention patching.

        # 3. Aggregate regional data
        regions =
        for i in range(1, 5):
            prompt = kwargs.get(f"region_{i}_prompt")
            mask = kwargs.get(f"region_{i}_mask")
            if prompt is not None and mask is not None:
                regions.append({"prompt": prompt, "mask": mask})

        if not regions:
            # No regions provided, so no patching is necessary.
            return (model, global_positive, global_negative, latent)

        # 4. Augment Conditioning
        # Positive conditioning
        positive_conds = [global_positive]
        positive_token_lengths = [global_positive.shape[1]]
        for region in regions:
            positive_conds.append(region["prompt"])
            positive_token_lengths.append(region["prompt"].shape[1])
        
        augmented_positive_cond = torch.cat(positive_conds, dim=1)
        augmented_positive = [[augmented_positive_cond, global_positive.[1]copy()]]

        # Negative conditioning (assuming global negative for all regions)
        negative_conds = [global_negative]
        for region in regions:
            # Use global negative for regional parts
            negative_conds.append(torch.zeros_like(region["prompt"]))
        
        augmented_negative_cond = torch.cat(negative_conds, dim=1)
        augmented_negative = [[augmented_negative_cond, global_negative.[1]copy()]]

        # 5. Process Masks and Prepare Attention Mask
        latent_height, latent_width = latent["samples"].shape[2], latent["samples"].shape[3]
        
        resized_masks =
        for region in regions:
            mask = region["mask"]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # Ensure mask is in the shape for interpolation
            resized_mask = F.interpolate(mask.unsqueeze(1), size=(latent_height, latent_width), mode="bilinear", align_corners=False)
            resized_masks.append(resized_mask.squeeze(1) > 0.5) # Binarize mask

        # 6. Build the Joint Attention Mask
        joint_attention_mask = self.create_joint_attention_mask(
            positive_token_lengths,
            (latent_height, latent_width),
            resized_masks,
            augmented_positive_cond.device
        )

        # 7. Define and Apply the Attention Patch
        # This wrapper function will replace the original attention calculation
        def regional_attention_wrapper(q, k, v, extra_options):
            # The wrapper uses the joint_attention_mask from the outer scope (closure)
            # This logic is inspired by similar regional attention nodes [2]
            
            # The shape of q, k, v is (Batch * Heads, SeqLen, Dim)
            # The mask needs to be broadcastable to (Batch * Heads, SeqLen, SeqLen)
            
            # We assume the mask is already prepared for the full sequence length
            # and just needs to be expanded for the batch and head dimensions.
            
            # The original function might be needed if we were to do more complex logic,
            # but here we replace it entirely with a call to scaled_dot_product_attention
            # which is the modern, efficient way to do this.
            
            # The mask should be additive (-inf for masked, 0 for not masked)
            # and broadcastable.
            
            # Check if the sequence length of q matches our mask
            if q.shape[1] == joint_attention_mask.shape:
                attn_mask = joint_attention_mask
            else:
                # This can happen if the negative prompt has a different length.
                # For simplicity, we don't apply regional control to negative prompts.
                attn_mask = None

            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        # Apply the patch using the robust ModelPatcher API [5]
        patched_model.set_model_attn1_patch(regional_attention_wrapper)
        patched_model.set_model_attn2_patch(regional_attention_wrapper)

        # 8. Clean up memory and return values
        del joint_attention_mask
        gc.collect()
        torch.cuda.empty_cache()

        return (patched_model, augmented_positive, augmented_negative, latent)

    def create_joint_attention_mask(self, token_lengths, latent_shape, region_masks, device):
        """
        Constructs the high-dimensional attention mask that enforces regional control.
        - Allows global prompt to attend to everything.
        - Allows each regional prompt to attend only to its own mask area in the latent space.
        - Blocks attention between different regional prompts.
        - Blocks attention between a regional prompt and other regions' latent areas.
        """
        latent_h, latent_w = latent_shape
        num_latent_tokens = latent_h * latent_w
        
        # Calculate start and end indices for each token group
        token_indices =
        current_index = 0
        for length in token_lengths:
            token_indices.append((current_index, current_index + length))
            current_index += length
        
        total_text_tokens = current_index
        total_tokens = total_text_tokens + num_latent_tokens
        
        # Initialize mask: 1 means block, 0 means allow
        mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)
        
        # Flatten region masks for easier indexing
        flat_region_masks = [m.view(-1) for m in region_masks]

        # Iterate through each region to apply constraints
        for i in range(len(region_masks)):
            # Region i's text tokens
            start_text_i, end_text_i = token_indices[i + 1] # +1 to skip global prompt
            
            # Region i's latent tokens
            latent_mask_i = flat_region_masks[i]
            latent_indices_i = torch.where(latent_mask_i) + total_text_tokens
            
            # Latent tokens NOT in region i
            latent_indices_not_i = torch.where(~latent_mask_i) + total_text_tokens

            # Rule 1: Block region i's text from attending to latents outside its mask
            mask[start_text_i:end_text_i, latent_indices_not_i] = 1
            mask[latent_indices_not_i, start_text_i:end_text_i] = 1
            
            # Iterate through other regions to block inter-entity attention
            for j in range(len(region_masks)):
                if i == j:
                    continue
                
                # Region j's text tokens
                start_text_j, end_text_j = token_indices[j + 1]
                
                # Region j's latent tokens
                latent_mask_j = flat_region_masks[j]
                latent_indices_j = torch.where(latent_mask_j) + total_text_tokens

                # Rule 2: Block attention between text tokens of region i and region j
                mask[start_text_i:end_text_i, start_text_j:end_text_j] = 1
                
                # Rule 3: Block region i's text from attending to region j's latents
                mask[start_text_i:end_text_i, latent_indices_j] = 1
                mask[latent_indices_j, start_text_i:end_text_i] = 1

        # Convert the boolean mask to the additive float mask required by scaled_dot_product_attention
        # Where the mask is 1 (block), the value becomes -inf.
        additive_mask = torch.zeros_like(mask, dtype=torch.float16)
        additive_mask.masked_fill_(mask, float('-inf'))
        
        return additive_mask

# Dictionary that ComfyUI uses to map node names to their classes
NODE_CLASS_MAPPINGS = {
    "EliGenRegionalControl": EliGenRegionalControl
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "EliGenRegionalControl": "EliGen Regional Control"
}
