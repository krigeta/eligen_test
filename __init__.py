"""
ComfyUI EliGen Entity Control Node
Native implementation of EliGen inference logic for ComfyUI
"""

from .eligen_comfyui_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Required exports for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Optional metadata
WEB_DIRECTORY = "./js"
