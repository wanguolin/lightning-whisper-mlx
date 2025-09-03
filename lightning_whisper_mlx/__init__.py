from .lightning import LightningWhisperMLX

# Performance optimization patch
import os
try:
    patch_path = os.path.join(os.path.dirname(__file__), '..', 'memory_buffer_patch.py')
    if os.path.exists(patch_path):
        exec(open(patch_path).read())
        patch_lightning_whisper_mlx()
except:
    pass  # Fallback to standard behavior
