import csv
import io
import numpy as np
import pickle
import requests
import json

import librosa
import pygit2
import gdown

# mega.py is optional due to dependency conflicts with Python 3.12+
try:
    from mega import Mega
    MEGA_AVAILABLE = True
except ImportError:
    MEGA_AVAILABLE = False


# =============================================================================
# R3GAN Model Registry
# =============================================================================
# R3GAN: A Modern GAN Baseline (NeurIPS 2024)
# Paper: https://arxiv.org/abs/2501.05441
# Code: https://github.com/brownvc/R3GAN

R3GAN_MODELS = {
    'r3gan_ffhq_256': {
        'name': 'r3gan_ffhq_256',
        'download_url': 'https://huggingface.co/brownvc/R3GAN-FFHQ-256x256/resolve/main/ffhq-256x256.pkl',
        'resolution': 256,
        'num_classes': 0,
        'type': 'r3gan',
        'description': 'R3GAN FFHQ 256x256 - High quality faces (FID: 2.75)',
    },
    'r3gan_ffhq_64': {
        'name': 'r3gan_ffhq_64',
        'download_url': 'https://huggingface.co/brownvc/R3GAN-FFHQ-64x64/resolve/main/ffhq-64x64.pkl',
        'resolution': 64,
        'num_classes': 0,
        'type': 'r3gan',
        'description': 'R3GAN FFHQ 64x64 - Fast face generation (FID: 1.95)',
    },
    'r3gan_cifar10': {
        'name': 'r3gan_cifar10',
        'download_url': 'https://huggingface.co/brownvc/R3GAN-CIFAR10/resolve/main/cifar10.pkl',
        'resolution': 32,
        'num_classes': 10,
        'type': 'r3gan',
        'description': 'R3GAN CIFAR-10 - 10-class conditional (FID: 1.96)',
    },
    'r3gan_imagenet_64': {
        'name': 'r3gan_imagenet_64',
        'download_url': 'https://huggingface.co/brownvc/R3GAN-ImgNet-64x64/resolve/main/imagenet-64x64.pkl',
        'resolution': 64,
        'num_classes': 1000,
        'type': 'r3gan',
        'description': 'R3GAN ImageNet 64x64 - 1000-class conditional (FID: 2.09)',
    },
    'r3gan_imagenet_32': {
        'name': 'r3gan_imagenet_32',
        'download_url': 'https://huggingface.co/brownvc/R3GAN-ImgNet-32x32/resolve/main/imagenet-32x32.pkl',
        'resolution': 32,
        'num_classes': 1000,
        'type': 'r3gan',
        'description': 'R3GAN ImageNet 32x32 - 1000-class conditional (FID: 1.27)',
    },
}


def get_r3gan_models():
    """Get list of available R3GAN models in the same format as consolidate_models()"""
    return list(R3GAN_MODELS.values())


def download_weights(url, output):
  '''Download model weights from URL'''

  if 'drive.google.com' in url:
    gdown.download(url, output=output, quiet=False)

  elif 'mega.nz' in url:
    if not MEGA_AVAILABLE:
      raise ImportError(
        "mega.py is required for downloading from mega.nz but is not installed. "
        "Install it with: pip install mega.py (note: requires Python < 3.12 due to dependency conflicts)"
      )
    m = Mega()
    m.login().download_url(url, dest_filename=output)

  elif 'yadi.sk' in url:
    endpoint = 'https://cloud-api.yandex.net/v1/disk/'\
               'public/resources/download?public_key='
    r_pre = requests.get(endpoint + url)
    r_pre_href = r_pre.json().get('href')
    r = requests.get(r_pre_href)
    with open(output, 'wb') as f:
      f.write(r.content)

  else:
    r = requests.get(url)
    with open(output, 'wb') as f:
      f.write(r.content)


# Cache for model list to avoid repeated network requests
_models_cache = None


def consolidate_models():
  '''Consolidate JSON dictionaries of pre-trained StyleGAN(2) and R3GAN weights'''
  global _models_cache

  # Return cached result if available
  if _models_cache is not None:
    return _models_cache

  # Define URL's for pre-trained StyleGAN and StyleGAN2 weights
  stylegan_url = 'https://raw.githubusercontent.com/justinpinkney/'\
  'awesome-pretrained-stylegan/master/models.csv'
  stylegan2_url = 'https://raw.githubusercontent.com/justinpinkney/'\
  'awesome-pretrained-stylegan2/master/models.json'

  # Load CSV without pandas - use stdlib csv module (faster import, same result)
  r_csv = requests.get(stylegan_url)
  csv_reader = csv.DictReader(io.StringIO(r_csv.text))
  models_stylegan = list(csv_reader)

  # Load JSON dictionary of StyleGAN2 weights
  r = requests.get(stylegan2_url)
  models_stylegan2 = json.loads(r.text)

  # Get R3GAN models
  models_r3gan = get_r3gan_models()

  # Consolidate StyleGAN, StyleGAN2, and R3GAN weights
  all_models = models_stylegan + models_stylegan2 + models_r3gan

  # Cache the result
  _models_cache = all_models

  return all_models


def is_r3gan_style(style):
  """Check if a style name refers to an R3GAN model"""
  if isinstance(style, str):
    return style.lower() in R3GAN_MODELS or style.lower().startswith('r3gan')
  return False


def get_spec_norm(wav, sr, n_mels, hop_length):
  '''Obtain maximum value for each time-frame in Mel Spectrogram, 
     and normalize between 0 and 1'''

  # Generate Mel Spectrogram
  spec_raw= librosa.feature.melspectrogram(y=wav, sr=sr,
                                           n_mels=n_mels,
                                           hop_length=hop_length)
  
  # Obtain maximum value per time-frame
  spec_max = np.amax(spec_raw,axis=0)

  # Normalize all values between 0 and 1
  spec_norm = (spec_max - np.min(spec_max))/np.ptp(spec_max)

  return spec_norm


def interpolate(array_1: np.ndarray, array_2: np.ndarray, steps: int):
  '''Linear interpolation between 2 arrays - vectorized version'''
  # Vectorized: compute all interpolations at once using broadcasting
  # Shape: (steps, 1) * (array_shape,) -> (steps, array_shape)
  t = np.linspace(0, 1, steps)[:, np.newaxis]
  return (1 - t) * array_1 + t * array_2


def full_frame_interpolation(frame_init, steps, len_output):
  '''Given a list of arrays (frame_init), produce linear interpolations between
     each pair of arrays. Optimized with pre-allocation.'''

  frame_init = np.asarray(frame_init)
  n_segments = len(frame_init) - 1

  if n_segments <= 0:
    # Single frame, just tile it
    return [frame_init[0]] * len_output

  # Pre-allocate output array
  total_frames = n_segments * steps
  frame_shape = frame_init[0].shape
  frames_array = np.empty((total_frames,) + frame_shape, dtype=frame_init.dtype)

  # Vectorized interpolation for each segment
  t = np.linspace(0, 1, steps)[:, np.newaxis]
  for i in range(n_segments):
    start_idx = i * steps
    end_idx = start_idx + steps
    frames_array[start_idx:end_idx] = (1 - t) * frame_init[i] + t * frame_init[i + 1]

  # Convert to list and pad if needed
  frames = list(frames_array)

  # Pad with final frame if needed (using list extend is faster than while loop)
  if len(frames) < len_output:
    frames.extend([frames[-1]] * (len_output - len(frames)))

  return frames[:len_output]