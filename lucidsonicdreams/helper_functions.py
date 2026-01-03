import csv
import io
import numpy as np
import pickle
import requests
import json

import librosa
import pygit2
import gdown
from mega import Mega


def download_weights(url, output):
  '''Download model weights from URL'''

  if 'drive.google.com' in url:
    gdown.download(url, output=output, quiet=False)

  elif 'mega.nz' in url:
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
  '''Consolidate JSON dictionaries of pre-trained StyleGAN(2) weights'''
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

  # Consolidate StyleGAN and StyleGAN2 weights
  all_models = models_stylegan + models_stylegan2

  # Cache the result
  _models_cache = all_models

  return all_models


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