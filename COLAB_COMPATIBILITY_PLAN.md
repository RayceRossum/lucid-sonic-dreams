# Lucid Sonic Dreams - Google Colab Compatibility Plan

## Overview

This plan outlines the changes needed to make lucid-sonic-dreams fully compatible with Google Colab. The changes address path handling, codec compatibility, memory management, error handling, and provide a ready-to-use Colab notebook.

---

## Issues Identified

### Critical Issues

1. **Audio Codec Fallback (main.py:830, 842)**
   - Problem: Uses `audio_codec='aac'` which requires ffmpeg with libfdk-aac
   - Impact: Colab's ffmpeg may not have AAC encoder, causing video write failures
   - Solution: Try AAC first, fallback to 'libmp3lame' or 'mp3'

2. **Duplicate Video Writing Code (main.py:827-853)**
   - Problem: Video write code appears twice - once in try block, once after finally
   - Impact: Second write will fail because temp files are deleted in finally block
   - Solution: Remove duplicate code after the try/finally block

3. **Temporary File Handling (main.py:808, 814)**
   - Problem: Writes 'tmp.mp4' and 'tmp.wav' to current directory
   - Impact: Can fill up Colab's limited disk space; path conflicts possible
   - Solution: Use `/tmp` directory or Python's `tempfile` module

4. **Relative Path for StyleGAN Repos (main.py:32-34, 107-109, 123-125)**
   - Problem: Clones StyleGAN to relative path 'stylegan2' or 'stylegan2_tf'
   - Impact: If working directory changes, imports fail
   - Solution: Use absolute paths based on package location or configurable base directory

### Medium Priority Issues

5. **sys.exit() Usage (main.py:82, 86, 144, 160, 703, 710, 715, 722, 872)**
   - Problem: Uses `sys.exit()` for error handling
   - Impact: Crashes Colab kernel instead of raising catchable exceptions
   - Solution: Replace with custom exception classes

6. **GPU Memory Management**
   - Problem: No explicit memory limits or monitoring
   - Impact: Can OOM on Colab's T4/K80 GPUs with large batch sizes or resolutions
   - Solution: Add memory estimation and warnings, suggest appropriate batch sizes

7. **MoviePy Codec Detection**
   - Problem: No check if ffmpeg has required codecs
   - Impact: Cryptic errors when codec unavailable
   - Solution: Add codec availability check at startup

### Low Priority Issues

8. **File Path Fragility (main.py:578)**
   - Problem: `file_name.split('.mp4')[0]` assumes specific naming
   - Impact: Fails with filenames containing multiple dots
   - Solution: Use `os.path.splitext()` or pathlib

9. **CUDA Benchmark Mode (main.py:28)**
   - Problem: `torch.backends.cudnn.benchmark = True` always enabled
   - Impact: May cause issues with variable input sizes
   - Solution: Make conditional based on use case

---

## Implementation Plan

### Phase 1: Core Fixes (main.py)

#### 1.1 Add Colab Environment Detection
```python
# Add to top of main.py
def is_colab():
    """Detect if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_temp_dir():
    """Get appropriate temp directory"""
    if is_colab():
        return '/tmp'
    return '.'
```

#### 1.2 Fix Audio Codec with Fallback
Replace lines 830 and 842:
```python
def write_video_with_fallback(video, file_name, audio_bitrate="1024k"):
    """Write video with codec fallback"""
    codecs = ['aac', 'libmp3lame', 'mp3']
    for codec in codecs:
        try:
            video.write_videofile(file_name, audio_codec=codec, audio_bitrate=audio_bitrate)
            return
        except Exception as e:
            if codec == codecs[-1]:
                raise RuntimeError(f"Failed to write video with any codec: {e}")
            continue
```

#### 1.3 Fix Temporary File Handling
Update temp file paths to use temp directory:
```python
temp_dir = get_temp_dir()
temp_video_path = os.path.join(temp_dir, 'lsd_tmp.mp4')
temp_audio_path = os.path.join(temp_dir, 'lsd_tmp.wav')
```

#### 1.4 Remove Duplicate Video Writing Code
Remove lines 841-853 (duplicate code after try/finally block)

#### 1.5 Fix StyleGAN Path Handling
Add configurable base directory for StyleGAN repos:
```python
def get_stylegan_dir():
    """Get StyleGAN directory based on environment"""
    if is_colab():
        return '/content/stylegan2'
    return os.path.join(os.path.dirname(__file__), '..', 'stylegan2')
```

### Phase 2: Exception Handling

#### 2.1 Create Custom Exceptions
```python
class LucidSonicDreamError(Exception):
    """Base exception for lucid-sonic-dreams"""
    pass

class StyleNotFoundError(LucidSonicDreamError):
    """Style not found in available styles"""
    pass

class InvalidParameterError(LucidSonicDreamError):
    """Invalid parameter value"""
    pass

class DownloadError(LucidSonicDreamError):
    """Failed to download model weights"""
    pass
```

#### 2.2 Replace sys.exit() Calls
Replace all `sys.exit()` calls with appropriate exception raises:
- Line 82-83: `raise InvalidParameterError(...)`
- Line 86-87: `raise InvalidParameterError(...)`
- Line 144-145: `raise StyleNotFoundError(...)`
- Line 160-161: `raise DownloadError(...)`
- etc.

### Phase 3: Memory Management

#### 3.1 Add GPU Memory Utilities
```python
def get_gpu_memory_mb():
    """Get available GPU memory in MB"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    return 0

def estimate_memory_usage(resolution, batch_size, input_shape=512):
    """Estimate GPU memory usage in MB"""
    # Rough estimation based on typical StyleGAN2 usage
    base_model = 1500  # ~1.5GB for model
    per_frame = (resolution ** 2 * 3 * 4) / 1024 / 1024  # RGB float32
    return base_model + (per_frame * batch_size * 2)  # 2x for forward/backward

def suggest_batch_size(resolution, available_memory_mb=None):
    """Suggest appropriate batch size for available memory"""
    if available_memory_mb is None:
        available_memory_mb = get_gpu_memory_mb()

    # Conservative estimates for Colab GPUs
    if available_memory_mb < 8000:  # < 8GB (K80)
        return 1
    elif available_memory_mb < 16000:  # < 16GB (T4)
        if resolution >= 1024:
            return 1
        return 2
    else:  # >= 16GB
        if resolution >= 1024:
            return 2
        return 4
```

### Phase 4: File Path Improvements

#### 4.1 Use pathlib for Path Handling
```python
from pathlib import Path

# Replace line 578
self.frames_dir = str(Path(file_name).stem) + '_frames'
```

### Phase 5: Create Colab Example Notebook

Create `examples/lucid_sonic_dreams_colab.ipynb` with:
1. Installation cell with all dependencies
2. Google Drive mounting (optional)
3. StyleGAN setup with symbolic links
4. Example usage with recommended parameters for Colab
5. Tips for memory management
6. Cleanup utilities

---

## Files to Modify

| File | Changes |
|------|---------|
| `lucidsonicdreams/main.py` | Core fixes (codec, paths, temp files, exceptions) |
| `lucidsonicdreams/helper_functions.py` | Add Colab utilities |
| `lucidsonicdreams/__init__.py` | Export new utilities |
| `setup.py` | Add imageio dependency (already imported but not in deps) |
| `examples/lucid_sonic_dreams_colab.ipynb` | NEW: Colab-ready notebook |

---

## Testing Checklist

- [ ] Video generation works on Colab with default parameters
- [ ] AAC codec fallback works when AAC unavailable
- [ ] Temp files are created in /tmp on Colab
- [ ] StyleGAN repos clone to correct location
- [ ] Custom exceptions are raised instead of sys.exit()
- [ ] Memory warnings appear for large resolutions
- [ ] All example code in notebook runs without errors
- [ ] Cleanup properly removes temp files

---

## Colab Notebook Structure

```
1. Setup & Installation
   - Clone repo
   - Install dependencies
   - Setup StyleGAN symlinks

2. Mount Google Drive (Optional)
   - For accessing models/audio from Drive

3. Quick Start Example
   - Simple 30-second video generation
   - Low resolution for fast testing

4. Full Example with Custom Parameters
   - Higher resolution
   - Custom effects
   - Multiple style options

5. Advanced: Custom Effects
   - Swirl effect example
   - Zoom effect example

6. Cleanup
   - Remove temp files
   - Clear GPU memory
```

---

## Recommended Colab Parameters

For **Colab Free** (T4 GPU, ~15GB RAM):
```python
resolution = 512      # Start low
batch_size = 1        # Conservative
fps = 30              # Standard
duration = 60         # 1 minute test
```

For **Colab Pro** (A100/V100, more RAM):
```python
resolution = 1024     # Higher quality
batch_size = 2        # Can handle more
fps = 60              # Smoother
duration = None       # Full song
```
