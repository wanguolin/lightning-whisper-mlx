import io
import numpy as np
import soundfile as sf
import tempfile

def transcribe_from_memory_buffer(audio_array, **kwargs):
    """
    Optimized transcription that uses in-memory buffer instead of file I/O.
    This reduces disk write operations and improves performance.
    """
    # Create in-memory WAV buffer
    buffer = io.BytesIO()
    
    # Write WAV data directly to memory buffer
    sf.write(buffer, audio_array, 16000, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    
    # Create a temporary file using the memory buffer content
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(buffer.getvalue())
        temp_file.flush()
        
        # Import the main transcribe function
        from lightning_whisper_mlx.transcribe import transcribe_audio
        
        # Call the main transcription function
        result = transcribe_audio(temp_file.name, **kwargs)
        
        # Clean up
        import os
        os.unlink(temp_file.name)
        
    return result

# Monkey patch the transcribe method to use memory buffer
def patch_lightning_whisper_mlx():
    try:
        from lightning_whisper_mlx import LightningWhisperMLX
        
        # Store original method
        original_transcribe = LightningWhisperMLX.transcribe
        
        def optimized_transcribe(self, audio_path, language=None, initial_prompt=None):
            # If audio_path is actually a numpy array, use memory buffer
            if isinstance(audio_path, np.ndarray):
                return transcribe_from_memory_buffer(
                    audio_path,
                    path_or_hf_repo=f'./mlx_models/{self.name}',
                    language=language,
                    batch_size=self.batch_size,
                    initial_prompt=initial_prompt
                )
            else:
                # Use original method for file paths
                return original_transcribe(self, audio_path, language, initial_prompt)
        
        # Replace the method
        LightningWhisperMLX.transcribe = optimized_transcribe
        print("✅ Successfully patched Lightning Whisper MLX for memory buffer support")
        
    except Exception as e:
        print(f"⚠️  Could not patch Lightning Whisper MLX: {e}")
        return False
    
    return True
