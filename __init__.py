# ComfyUI-EliGen/eligen_nodes.py

import torch
import comfy.sample
import comfy.samplers
from comfy.model_patcher import ModelPatcher

class EliGenEntityDefinition:
    """
    Node to define a single entity with its prompt and mask.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "entity_prompt": ("STRING", {"multiline": True, "default": "a red apple"}),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("ELIGEN_ENTITY",)
    FUNCTION = "define_entity"
    CATEGORY = "EliGen"

    def define_entity(self, entity_prompt, mask):
        # The mask tensor from ComfyUI is. We only need one channel.
        if mask.dim() == 3:
            mask = mask[:, :, 0]
        # Package the prompt and mask into a list containing a dictionary.
        # This structure allows for easy combination later.
        entity_package = [{"prompt": entity_prompt, "mask": mask}]
        return (entity_package,)

class EliGenEntityPacker:
    """
    Node to aggregate multiple entities and the global prompt into a single payload.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive": ("CONDITIONING",),
                "entities": ("ELIGEN_ENTITY",),
            }
        }

    RETURN_TYPES = ("ELIGEN_PAYLOAD",)
    FUNCTION = "pack_entities"
    CATEGORY = "EliGen"

    def pack_entities(self, clip, positive, entities):
        # The 'entities' input is a list of lists of dictionaries. Flatten it.
        flat_entities = [item for sublist in entities for item in sublist]

        # Extract the global conditioning tensor and pooled output
        global_cond, global_pooled = positive

        entity_conds =
        entity_masks =

        # Tokenize and encode each entity prompt
        for entity in flat_entities:
            tokens = clip.tokenize(entity["prompt"])
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            entity_conds.append(cond)
            entity_masks.append(entity["mask"])

        # Create the final payload dictionary
        payload = {
            "global_cond": global_cond,
            "global_pooled": global_pooled,
            "entity_conds": entity_conds,
            "entity_masks": entity_masks
        }
        return (payload,)

#... imports from above...

class EliGenAttentionPatcher:
    """
    A context manager to temporarily patch the model's attention mechanism.
    """
    def __init__(self, model, payload):
        self.model = model
        self.payload = payload
        self.original_methods = {}

    def __enter__(self):
        # Find all CrossAttention modules and patch them
        for name, module in self.model.model.named_modules():
            if "CrossAttention" in module.__class__.__name__:
                self.original_methods[name] = module.forward
                module.forward = self.forward_eligen.__get__(module, module.__class__)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original methods
        for name, module in self.model.model.named_modules():
            if name in self.original_methods:
                module.forward = self.original_methods[name]

    def forward_eligen(self, x, context=None, mask=None):
        """
        The custom forward pass with regional attention logic.
        'self' here refers to the instance of the CrossAttention module.
        """
        # Standard attention setup
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        q, k, v = map(lambda t: t.view(t.shape, -1, self.heads, self.dim_head).transpose(1, 2), (q, k, v))

        # --- EliGen Regional Attention Logic ---
        # Get the global and entity conditioning tensors from the payload
        global_cond = self.payload["global_cond"]
        entity_conds = self.payload["entity_conds"]
        
        # Concatenate all conditioning tensors to form the full key/value context
        full_context_k = torch.cat([global_cond] + entity_conds, dim=1)
        full_context_v = torch.cat([global_cond] + entity_conds, dim=1) # Assuming same tensor for k and v context
        
        # Recalculate k and v with the full context
        k = self.to_k(full_context_k)
        v = self.to_v(full_context_v)
        k, v = map(lambda t: t.view(t.shape, -1, self.heads, self.dim_head).transpose(1, 2), (k, v))
        
        # Calculate attention scores
        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Build the regional attention mask
        regional_mask = self.build_regional_mask(q, global_cond, entity_conds)
        
        # Apply the regional mask to the scores
        scores = scores + regional_mask

        # Continue with standard attention
        attn = scores.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).reshape(out.shape, -1, self.heads * self.dim_head)
        return self.to_out(out)

    def build_regional_mask(self, q, global_cond, entity_conds):
        """
        Constructs the mask to enforce regional attention.
        """
        # Get dimensions
        num_patches = q.shape
        latent_dim = int(num_patches**0.5)
        
        # Initialize mask with zeros (allowing all attention)
        mask = torch.zeros(1, 1, num_patches, global_cond.shape + sum(c.shape for c in entity_conds), device=q.device)
        
        # Get entity masks from payload and resize them to the latent dimensions
        entity_masks = [torch.nn.functional.interpolate(m.unsqueeze(0).unsqueeze(0), size=(latent_dim, latent_dim), mode='bilinear').squeeze() for m in self.payload["entity_masks"]]
        
        current_token_idx = global_cond.shape
        for i, cond in enumerate(entity_conds):
            num_tokens = cond.shape
            # Flatten the spatial mask to match the query tokens
            spatial_mask = entity_masks[i].flatten() > 0.5
            
            # For the tokens of this entity, set attention to -inf for all patches NOT in its mask
            mask[:, :, ~spatial_mask, current_token_idx:current_token_idx + num_tokens] = -torch.inf
            
            # For the patches within this entity's mask, set attention to -inf for all OTHER entity tokens
            # This prevents entities from "bleeding" into each other's regions
            other_entities_start = global_cond.shape
            for j, other_cond in enumerate(entity_conds):
                if i!= j:
                    other_num_tokens = other_cond.shape
                    mask[:, :, spatial_mask, other_entities_start:other_entities_start + other_num_tokens] = -torch.inf
                other_entities_start += other_cond.shape

            current_token_idx += num_tokens
            
        return mask

class EliGenSampler:
    """
    The main sampler node that orchestrates the EliGen process.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("ELIGEN_PAYLOAD",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "EliGen"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        # The 'positive' input is our custom payload dictionary
        eligen_payload = positive
        
        # The model's forward pass needs the pooled output. We combine the global one with the first entity's.
        # This is a simplification; more advanced handling might average them or use a dedicated mechanism.
        combined_pooled = eligen_payload["global_pooled"]
        
        # Prepare the conditioning list for the standard sampler function.
        # We need to format the global prompt with its pooled output for the sampler.
        positive_for_sampler = [[eligen_payload["global_cond"], {"pooled_output": combined_pooled}]]
        
        # Create a patched version of the model using our context manager
        patched_model = ModelPatcher(model.model, target_device=comfy.model_management.get_torch_device())
        
        with EliGenAttentionPatcher(patched_model, eligen_payload):
            # Call the standard ComfyUI sampler function, but with our patched model
            samples = comfy.sample.sample(patched_model, seed, steps, cfg, sampler_name, scheduler,
                                          positive_for_sampler, negative, latent_image,
                                          denoise=denoise)
        
        return (samples,)
