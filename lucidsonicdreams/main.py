import sys
import os
import shutil
import pickle
import tempfile
from pathlib import Path
from tqdm import tqdm
import inspect
import numpy as np
import random
from scipy.stats import truncnorm

import torch
import PIL
from PIL import Image, ImageEnhance
import skimage.exposure
import librosa
import soundfile
# MoviePy 2.x compatible imports
from moviepy import VideoFileClip, ImageSequenceClip, AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import pygit2
from importlib import import_module

from .helper_functions import *
from .helper_functions import is_r3gan_style, R3GAN_MODELS
from .sample_effects import *

import imageio

# For speed
torch.backends.cudnn.benchmark = True


# =============================================================================
# Custom Exceptions
# =============================================================================

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


# =============================================================================
# Colab Compatibility Utilities
# =============================================================================

def is_colab():
    """Detect if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_temp_dir():
    """Get appropriate temp directory based on environment"""
    if is_colab():
        return '/tmp'
    return tempfile.gettempdir()


def get_stylegan_base_dir():
    """Get base directory for StyleGAN repositories"""
    if is_colab():
        return '/content'
    # Use directory relative to this module
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# StyleGAN3 repo URL (preferred over StyleGAN2 - better maintained, same API)
STYLEGAN3_REPO = 'https://github.com/NVlabs/stylegan3.git'
STYLEGAN2_REPO = 'https://github.com/NVlabs/stylegan2-ada-pytorch.git'

# R3GAN repo URL - Modern GAN baseline with better FID scores
# Paper: https://arxiv.org/abs/2501.05441
R3GAN_REPO = 'https://github.com/brownvc/R3GAN.git'


def setup_stylegan(base_dir, use_stylegan3=True):
    """Setup StyleGAN repository with proper path handling.

    Args:
        base_dir: Base directory to clone into
        use_stylegan3: If True, use StyleGAN3 (recommended). Falls back to SG2 if needed.

    Returns:
        Path to the StyleGAN directory
    """
    sg3_path = os.path.join(base_dir, 'stylegan3')
    sg2_path = os.path.join(base_dir, 'stylegan2')

    # Prefer StyleGAN3
    if use_stylegan3:
        if not os.path.exists(sg3_path):
            print("Cloning StyleGAN3 repository...")
            pygit2.clone_repository(STYLEGAN3_REPO, sg3_path)

        # Create symlink for backwards compatibility
        if not os.path.exists(sg2_path):
            try:
                os.symlink(sg3_path, sg2_path)
            except OSError:
                pass  # Symlink may fail on some systems, that's OK

        return sg3_path
    else:
        if not os.path.exists(sg2_path):
            print("Cloning StyleGAN2-ada-pytorch repository...")
            pygit2.clone_repository(STYLEGAN2_REPO, sg2_path)
        return sg2_path


def setup_r3gan(base_dir):
    """Setup R3GAN repository with proper path handling.

    R3GAN is a modern GAN baseline that achieves state-of-the-art results
    while being simpler than StyleGAN2/3. It uses a regularized relativistic
    GAN loss (RpGAN + R1 + R2) with mathematically proven convergence.

    Paper: "The GAN is dead; long live the GAN! A Modern Baseline GAN"
    https://arxiv.org/abs/2501.05441

    Args:
        base_dir: Base directory to clone into

    Returns:
        Path to the R3GAN directory
    """
    r3gan_path = os.path.join(base_dir, 'R3GAN')

    if not os.path.exists(r3gan_path):
        print("Cloning R3GAN repository...")
        pygit2.clone_repository(R3GAN_REPO, r3gan_path)

    return r3gan_path


def get_gpu_memory_mb():
    """Get total GPU memory in MB, or 0 if no GPU"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    return 0


def get_free_gpu_memory_mb():
    """Get free GPU memory in MB, or 0 if no GPU"""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / 1024 / 1024
    return 0


def estimate_memory_usage_mb(resolution, batch_size):
    """Estimate GPU memory usage in MB for given parameters"""
    # Rough estimation based on typical StyleGAN2 usage
    base_model = 1500  # ~1.5GB for model weights
    per_frame = (resolution ** 2 * 3 * 4) / 1024 / 1024  # RGB float32
    return base_model + (per_frame * batch_size * 3)  # 3x for intermediate activations


def suggest_batch_size(resolution, available_memory_mb=None):
    """Suggest appropriate batch size for available GPU memory"""
    if available_memory_mb is None:
        available_memory_mb = get_gpu_memory_mb()

    if available_memory_mb == 0:
        return 1  # CPU mode

    # Conservative estimates for common GPU configurations
    if available_memory_mb < 8000:  # < 8GB (K80, older GPUs)
        return 1
    elif available_memory_mb < 16000:  # < 16GB (T4, RTX 3080)
        if resolution >= 1024:
            return 1
        elif resolution >= 512:
            return 2
        return 4
    else:  # >= 16GB (A100, V100, RTX 4090)
        if resolution >= 1024:
            return 2
        elif resolution >= 512:
            return 4
        return 8


def write_video_with_codec_fallback(video, file_name, audio_bitrate="1024k"):
    """Write video file with automatic codec fallback for compatibility"""
    codecs = ['aac', 'libmp3lame', 'mp3']
    last_error = None

    for codec in codecs:
        try:
            video.write_videofile(file_name, audio_codec=codec, audio_bitrate=audio_bitrate)
            return  # Success
        except Exception as e:
            last_error = e
            if codec != codecs[-1]:
                print(f"Codec '{codec}' failed, trying next...")
            continue

    raise RuntimeError(f"Failed to write video with any codec. Last error: {last_error}")

def import_stylegan_torch():
    # Clone Official StyleGAN2-ADA-pytorch Repository
    if not os.path.exists('stylegan2'):
        pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada-pytorch.git',
                              'stylegan2')
    # StyleGan2 imports
    sys.path.append("stylegan2")
    import legacy
    import dnnlib


def import_stylegan_tf():
    print("Cloning old, tensorflow stylegan...")
    if not os.path.exists('stylegan2_tf'):
        pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada.git',
                              'stylegan2_tf')

    #StyleGAN2 Imports
    sys.path.append("stylegan2_tf")
    import dnnlib as dnnlib
    from dnnlib.tflib.tfutil import convert_images_to_uint8 as convert_images_to_uint8
    init_tf()


def show_styles():
    '''Show names of available (non-custom) styles'''

    all_models = consolidate_models()
    styles = set([model['name'].lower() for model in all_models])
    print(*sorted(styles), sep='\n')


def show_r3gan_styles():
    '''Show available R3GAN model styles with descriptions.

    R3GAN (NeurIPS 2024) is a modern GAN baseline that achieves
    state-of-the-art FID scores with a simpler architecture than StyleGAN.

    Paper: "The GAN is dead; long live the GAN! A Modern Baseline GAN"
    https://arxiv.org/abs/2501.05441
    '''
    print("Available R3GAN Models:")
    print("=" * 60)
    for name, info in R3GAN_MODELS.items():
        print(f"\n  {name}")
        print(f"    Resolution: {info['resolution']}x{info['resolution']}")
        print(f"    Classes: {info['num_classes']}")
        print(f"    {info['description']}")
    print("\n" + "=" * 60)
    print("Usage: LucidSonicDream(song='song.mp3', style='r3gan_ffhq_256')")


class LucidSonicDream:
  def __init__(self,
               song: str,
               pulse_audio: str = None,
               motion_audio: str = None,
               class_audio: str = None,
               contrast_audio: str = None,
               flash_audio: str = None,
               style: str = 'wikiart',
               input_shape: int = None,
               num_possible_classes: int = None):

    # If style is a function, raise exception if function does not take
    # noise_batch or class_batch parameters
    if callable(style):

        func_sig = list(inspect.getfullargspec(style))[0]

        for arg in ['noise_batch', 'class_batch']:
            if arg not in func_sig:
                raise InvalidParameterError(
                    'func must be a function with parameters noise_batch and class_batch'
                )

        # Raise exception if input_shape or num_possible_classes is not provided
        if (input_shape is None) or (num_possible_classes is None):
            raise InvalidParameterError(
                'input_shape and num_possible_classes must be provided if style is a function'
            )

    # Define attributes
    self.song = song
    self.pulse_audio = pulse_audio
    self.motion_audio = motion_audio
    self.class_audio = class_audio
    self.contrast_audio = contrast_audio
    self.flash_audio = flash_audio
    self.style = style
    self.input_shape = input_shape or 512
    self.num_possible_classes = num_possible_classes
    self.style_exists = False

    # Get base directory for StyleGAN repos (Colab-compatible)
    stylegan_base = get_stylegan_base_dir()

    # Performance options
    self.use_fp16 = torch.cuda.is_available()  # Use FP16 on GPU for speed
    self.use_compile = hasattr(torch, 'compile')  # PyTorch 2.0+ compilation

    # Detect model type
    self.use_r3gan = is_r3gan_style(style) if isinstance(style, str) else False
    self.use_tf = style in ("wikiart",) if isinstance(style, str) else False

    if self.use_tf:
        stylegan_tf_path = os.path.join(stylegan_base, 'stylegan2_tf')
        print("Cloning old, tensorflow stylegan...")
        if not os.path.exists(stylegan_tf_path):
            pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada.git',
                                  stylegan_tf_path)

        # StyleGAN2 Imports
        if stylegan_tf_path not in sys.path:
            sys.path.insert(0, stylegan_tf_path)
        self.dnnlib = import_module("dnnlib")
        tflib = import_module("dnnlib.tflib.tfutil")
        self.convert_images_to_uint8 = tflib.convert_images_to_uint8
        self.init_tf = tflib.init_tf
        self.init_tf()
    elif self.use_r3gan:
        # Setup R3GAN repository
        r3gan_path = setup_r3gan(stylegan_base)
        # Clear any cached stylegan modules that would interfere with R3GAN
        # This is critical because R3GAN models need training.networks_baseline
        sys.path = [p for p in sys.path if 'stylegan' not in p.lower()]
        sys.path.insert(0, r3gan_path)
        # Force reimport - clear cached modules from stylegan3
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith(('training', 'dnnlib', 'legacy', 'torch_utils')):
                del sys.modules[mod_name]
        self.dnnlib = import_module("dnnlib")
        self.legacy = import_module("legacy")
        print("Using R3GAN - Modern GAN baseline (arxiv:2501.05441)")
    else:
        # Use StyleGAN3 (preferred) - backwards compatible with SG2 models
        stylegan_path = setup_stylegan(stylegan_base, use_stylegan3=True)
        if stylegan_path not in sys.path:
            sys.path.insert(0, stylegan_path)
        self.dnnlib = import_module("dnnlib")
        self.legacy = import_module("legacy")
    

  def stylegan_init(self):
    '''Initialise StyleGAN2-ada-pytorch or R3GAN weights'''

    style = self.style

    # If style is not a .pkl file path, download weights from corresponding URL
    if '.pkl' not in style:
      # Check if it's an R3GAN model first (faster lookup)
      style_lower = style.lower()
      if style_lower in R3GAN_MODELS:
        model_info = R3GAN_MODELS[style_lower]
        download_url = model_info['download_url']
        weights_file = style + '.pkl'
        print(f"R3GAN model: {model_info.get('description', style)}")
      else:
        # Fall back to consolidated models list
        all_models = consolidate_models()
        all_styles = [model['name'].lower() for model in all_models]

        # Raise exception if style is not valid
        if style_lower not in all_styles:
          raise StyleNotFoundError(
              'Style not valid. Call show_styles() to see all '
              'valid styles, or use your own .pkl file.'
          )

        download_url = [model for model in all_models \
                        if model['name'].lower() == style_lower][0]\
                        ['download_url']
        weights_file = style + '.pkl'

      # If style .pkl already exists in working directory, skip download
      if not os.path.exists(weights_file):
        print('Downloading {} weights (This may take a while)...'.format(style))
        try:
          download_weights(download_url, weights_file)
        except Exception as e:
          raise DownloadError(
              f'Download failed. Try to download weights directly at {download_url} '
              f'and pass the file path to the style parameter. Error: {e}'
          )
        print('Download complete')

    else:
      weights_file = style

    # load generator
    if self.use_tf:
        # Load weights
        with open(weights_file, 'rb') as f:
            self.Gs = pickle.load(f)[2]
    else:
        print(f'Loading networks from {weights_file}...')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For R3GAN models, ensure the R3GAN path is at the front of sys.path
        # so that training.networks_baseline can be found during pickle unpickling
        if self.use_r3gan:
            stylegan_base = get_stylegan_base_dir()
            r3gan_path = os.path.join(stylegan_base, 'R3GAN')
            # Remove any existing R3GAN path entries and add to front
            sys.path = [p for p in sys.path if 'R3GAN' not in p and 'stylegan' not in p.lower()]
            sys.path.insert(0, r3gan_path)
            # Force reimport of modules that may have been cached from stylegan3
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith('training') or mod_name in ('legacy', 'dnnlib'):
                    del sys.modules[mod_name]
            # Reimport with correct path
            self.dnnlib = import_module("dnnlib")
            self.legacy = import_module("legacy")

        with self.dnnlib.util.open_url(weights_file) as f:
            self.Gs = self.legacy.load_network_pkl(f)['G_ema'].to(self.device)

        # Performance optimizations
        self.Gs.eval()  # Set to evaluation mode

        # For R3GAN, prefer BFloat16 over FP16 (per the paper's recommendations)
        # R3GAN uses BFloat16 for mixed precision training
        if self.use_fp16 and self.device.type == 'cuda':
            try:
                if self.use_r3gan and hasattr(torch, 'bfloat16'):
                    # R3GAN prefers BFloat16
                    self.Gs = self.Gs.to(torch.bfloat16)
                    print("Using BFloat16 precision for R3GAN inference")
                else:
                    self.Gs = self.Gs.half()
                    print("Using FP16 precision for faster inference")
            except Exception as e:
                print(f"Mixed precision not supported for this model: {e}")
                self.use_fp16 = False

        # torch.compile for PyTorch 2.0+ (can provide 20-50% speedup)
        if self.use_compile and self.device.type == 'cuda':
            try:
                self.Gs = torch.compile(self.Gs, mode='reduce-overhead')
                print("Using torch.compile() for optimized inference")
            except Exception as e:
                print(f"torch.compile() not available: {e}")
                self.use_compile = False

    # Auto assign num_possible_classes attribute
    try:
      # Try StyleGAN3/R3GAN format first
      if hasattr(self.Gs, 'c_dim'):
        self.num_possible_classes = self.Gs.c_dim
      elif hasattr(self.Gs, 'mapping') and hasattr(self.Gs.mapping, 'input_templates'):
        print(self.Gs.mapping.input_templates)
        self.num_possible_classes = self.Gs.mapping.input_templates[1].shape[1]
      else:
        self.num_possible_classes = 0
    except ValueError:
      try:
        print(self.Gs.mapping.static_kwargs.label_size)
        self.num_possible_classes = self.Gs.components.mapping\
                                    .static_kwargs.label_size
      except Exception:
        self.num_possible_classes = 0
    except Exception:
      self.num_possible_classes = 0      


  def load_specs(self):
    '''Load normalized spectrograms and chromagram'''

    start = self.start
    duration = self.duration
    fps = self.fps
    input_shape = self.input_shape
    pulse_percussive = self.pulse_percussive
    pulse_harmonic = self.pulse_harmonic
    motion_percussive = self.motion_percussive
    motion_harmonic = self.motion_harmonic

    # Load audio signal data. Now with dirty hax for rounded fps values!
    if 24<= fps <=60 :
      # fps hack
      sample_rate = 512 * fps
    else:
      # the default librosa sample rate
      sample_rate = 22050
    wav, sr = librosa.load(self.song, sr=sample_rate, offset=start, duration=duration)
    wav_motion = wav_pulse = wav_class = wav
    sr_motion = sr_pulse = sr_class = sr

    # If pulse_percussive != pulse_harmonic
    # or motion_percussive != motion_harmonic,
    # decompose harmonic and percussive signals and assign accordingly
    aud_unassigned = (not self.pulse_audio) or (not self.motion_audio)
    pulse_bools_equal = pulse_percussive == pulse_harmonic
    motion_bools_equal = motion_percussive == motion_harmonic

    if aud_unassigned and not all([pulse_bools_equal, motion_bools_equal]):
       wav_harm, wav_perc = librosa.effects.hpss(wav)
       wav_list = [wav, wav_harm, wav_perc]

       pulse_bools = [pulse_bools_equal, pulse_harmonic, pulse_percussive]
       wav_pulse = wav_list[pulse_bools.index(max(pulse_bools))]

       motion_bools = [motion_bools_equal, motion_harmonic, motion_percussive]
       wav_motion = wav_list[motion_bools.index(max(motion_bools))]

    # Load audio signal data for Pulse, Motion, and Class if provided
    if self.pulse_audio:
      wav_pulse, sr_pulse = librosa.load(self.pulse_audio, sr=sample_rate,
                                        offset=start, duration=duration)
    if self.motion_audio:
      wav_motion, sr_motion = librosa.load(self.motion_audio, sr=sample_rate,
                                        offset=start, duration=duration)
    if self.class_audio:
      wav_class, sr_class = librosa.load(self.class_audio, sr=sample_rate,
                                        offset=start, duration=duration)
    
    # Calculate frame duration (i.e. samples per frame)
    frame_duration = int(sr/fps - (sr/fps % 64))

    # Generate normalized spectrograms for Pulse, Motion and Class
    self.spec_norm_pulse = get_spec_norm(wav_pulse, sr_pulse, 
                                         input_shape, frame_duration)
    self.spec_norm_motion = get_spec_norm(wav_motion, sr_motion,
                                          input_shape, frame_duration)
    self.spec_norm_class= get_spec_norm(wav_class,sr_class, 
                                        input_shape, frame_duration)

    # Generate chromagram from Class audio
    chrom_class = librosa.feature.chroma_cqt(y=wav_class, sr=sr,
                                             hop_length=frame_duration)
    # Sort pitches based on "dominance"
    chrom_class_norm = chrom_class/\
                       chrom_class.sum(axis = 0, keepdims = 1)
    chrom_class_sum = np.sum(chrom_class_norm,axis=1)
    pitches_sorted = np.argsort(chrom_class_sum)[::-1]

    # Assign attributes to be used for vector generation
    self.wav, self.sr, self.frame_duration = wav, sr, frame_duration
    self.chrom_class, self.pitches_sorted = chrom_class, pitches_sorted


  def transform_classes(self):
    '''Transform/assign value of classes'''
    print("Number of model classes: ", self.num_possible_classes)
    # If model does not use classes, simply return list of 0's
    if self.num_possible_classes == 0:
      self.classes = [0]*12

    else:

      # If list of classes is not provided, generate a random sample
      if self.classes is None: 
        self.classes = random.sample(range(self.num_possible_classes),
                                     min([self.num_possible_classes,12]))
      
      # If length of list < 12, repeat list until length is 12
      if len(self.classes) < 12:
        self.classes = (self.classes * int(np.ceil(12/len(self.classes))))[:12]

      # If dominant_classes_first is True, sort classes accordingly  
      if self.dominant_classes_first:
        self.classes=[self.classes[i] for i in np.argsort(self.pitches_sorted)]


  def update_motion_signs(self):
    '''Update direction of noise interpolation based on truncation value'''
    m = self.motion_react
    t = self.truncation
    motion_signs = self.motion_signs
    current_noise = self.current_noise

    # Vectorized version - much faster than np.vectorize which is just a loop
    # For each current value in noise vector, change direction if absolute
    # value +/- motion_react is larger than 2*truncation
    result = motion_signs.copy()
    result[current_noise - m < -2*t] = 1
    result[current_noise + m >= 2*t] = -1
    return result

  def generate_class_vec(self, frame):
    '''Generate a class vector using chromagram, where each pitch
       corresponds to a class'''

    classes = self.classes
    chrom_class = self.chrom_class
    class_vecs = self.class_vecs
    num_possible_classes = self.num_possible_classes
    class_complexity = self.class_complexity
    class_pitch_react = self.class_pitch_react * 43 / self.fps

    # Pre-allocate class vector with zeros
    class_vec = np.zeros(num_possible_classes, dtype=np.float32)

    # For the first class vector, use values from the first point in time
    # where at least one pitch > 0 (controls for silence at the start)
    if len(class_vecs) == 0:
      col_sums = chrom_class.sum(axis=0)
      nonzero_cols = np.where(col_sums > 0)[0]
      if len(nonzero_cols) > 0:
        first_chrom = chrom_class[:, nonzero_cols[0]]
      else:
        first_chrom = chrom_class[:, 0]

      # Vectorized assignment using numpy indexing
      valid_classes = np.array(classes[:len(first_chrom)])
      valid_mask = valid_classes < num_possible_classes
      class_vec[valid_classes[valid_mask]] = first_chrom[valid_mask]

    # For succeeding vectors, update class values scaled by class_pitch_react
    else:
      chrom_values = chrom_class[:, frame]
      update_vec = np.zeros(num_possible_classes, dtype=np.float32)

      # Vectorized assignment
      valid_classes = np.array(classes[:len(chrom_values)])
      valid_mask = valid_classes < num_possible_classes
      update_vec[valid_classes[valid_mask]] = chrom_values[valid_mask]

      class_vec = class_vecs[frame - 1] + class_pitch_react * update_vec

    # Normalize class vector between 0 and 1
    nonzero_mask = class_vec != 0
    if np.any(nonzero_mask):
      positive_vals = class_vec[class_vec >= 0]
      if len(positive_vals) > 0:
        class_vec[class_vec < 0] = np.min(positive_vals)
      ptp = np.ptp(class_vec)
      if ptp > 0:
        class_vec = (class_vec - np.min(class_vec)) / ptp

    # If all values in class vector are equal, add 0.1 to first value
    if num_possible_classes > 0 and np.all(class_vec == class_vec[0]):
      class_vec[0] += 0.1

    return class_vec * class_complexity
            

  def is_shuffle_frame(self, frame):
    '''Determines if classes should be shuffled in current frame'''

    class_shuffle_seconds = self.class_shuffle_seconds 
    fps = self.fps 

    # If class_shuffle_seconds is an integer, return True if current timestamp
    # (in seconds) is divisible by this integer
    if type(class_shuffle_seconds) == int:
      if frame != 0 and frame % round(class_shuffle_seconds*fps) == 0:
        return True
      else:
        return False 

    # If class_shuffle_seconds is a list, return True if current timestamp 
    # (in seconds) is in list
    if type(class_shuffle_seconds) == list:
      if frame/fps + self.start in class_shuffle_seconds:
        return True
      else:
        return False


  def generate_vectors(self):
    '''Generates noise and class vectors as inputs for each frame'''

    PULSE_SMOOTH = 0.75
    MOTION_SMOOTH = 0.75
    classes = self.classes
    class_shuffle_seconds = self.class_shuffle_seconds or [0]
    class_shuffle_strength = round(self.class_shuffle_strength * 12)
    fps = self.fps
    class_smooth_frames = int(self.class_smooth_seconds * fps)
    motion_react = self.motion_react * 20 / fps

    # Get number of noise vectors to initialise (based on speed_fpm)
    num_init_noise = round(
      librosa.get_duration(path=self.wav, sr=self.sr)/60*self.speed_fpm)

    num_frames = len(self.spec_norm_class)

    # If num_init_noise < 2, simply initialise the same
    # noise vector for all frames
    if num_init_noise < 2:
      base_noise = self.truncation * \
               truncnorm.rvs(-2, 2,
                             size=(self.batch_size, self.input_shape)) \
                        .astype(np.float32)[0]
      noise = [base_noise.copy() for _ in range(num_frames)]

    # Otherwise, initialise num_init_noise different vectors, and generate
    # linear interpolations between these vectors
    else:
      # Vectorized initialization of noise vectors
      init_noise = self.truncation * \
                   truncnorm.rvs(-2, 2,
                                 size=(num_init_noise, self.input_shape)) \
                            .astype(np.float32)

      # Compute number of steps between each pair of vectors
      steps = int(np.floor(num_frames) / len(init_noise) - 1)

      # Interpolate
      noise = full_frame_interpolation(init_noise,
                                       steps,
                                       num_frames)

    # Pre-allocate arrays instead of using append (much faster)
    pulse_noise = np.empty((num_frames, self.input_shape), dtype=np.float32)
    motion_noise = np.empty((num_frames, self.input_shape), dtype=np.float32)
    self.class_vecs = []

    # Use np.full instead of np.array([x]*n) - faster
    pulse_base = np.full(self.input_shape, self.pulse_react, dtype=np.float32)
    motion_base = np.full(self.input_shape, motion_react, dtype=np.float32)

    # Use np.random.choice instead of list comprehension - vectorized
    self.motion_signs = np.random.choice([1, -1], size=self.input_shape).astype(np.float32)

    # Vectorized random factors
    rand_choices = np.array([1, 1 - self.motion_randomness], dtype=np.float32)
    rand_factors = np.random.choice(rand_choices, size=self.input_shape)

    # Pre-compute refresh interval
    refresh_interval = max(1, round(fps * 4))

    cumm_motion_noise = np.zeros(self.input_shape, dtype=np.float32)

    for i in range(num_frames):

      # UPDATE NOISE #

      # Re-initialise randomness factors every 4 seconds (vectorized)
      if i % refresh_interval == 0:
        rand_factors = np.random.choice(rand_choices, size=self.input_shape)

      # Generate incremental update vectors for Pulse and Motion
      pulse_noise_add = pulse_base * self.spec_norm_pulse[i]
      motion_noise_add = motion_base * self.spec_norm_motion[i] * \
                         self.motion_signs * rand_factors

      # Smooth each update vector using a weighted average of
      # itself and the previous vector
      if i > 0:
        pulse_noise_add = pulse_noise[i-1] * PULSE_SMOOTH + \
                          pulse_noise_add * (1 - PULSE_SMOOTH)
        motion_noise_add = motion_noise[i-1] * MOTION_SMOOTH + \
                           motion_noise_add * (1 - MOTION_SMOOTH)

      # Store in pre-allocated arrays
      pulse_noise[i] = pulse_noise_add
      motion_noise[i] = motion_noise_add
      cumm_motion_noise += motion_noise_add

      # Update current noise vector
      noise[i] = noise[i] + pulse_noise_add + cumm_motion_noise

      self.current_noise = noise[i]

      # Update directions
      self.motion_signs = self.update_motion_signs()

      # UPDATE CLASSES #

      # If current frame is a shuffle frame, shuffle classes accordingly
      if self.is_shuffle_frame(i):
        self.classes = self.classes[class_shuffle_strength:] + \
                       self.classes[:class_shuffle_strength]

      # Generate class update vector and append to list
      class_vec_add = self.generate_class_vec(frame=i)
      self.class_vecs.append(class_vec_add)

    # Store final noise
    self.noise = noise

    # Smoothen class vectors by obtaining the mean vector per
    # class_smooth_frames frames, and interpolating between these vectors
    if class_smooth_frames > 1:
      # Convert to numpy array for faster operations
      class_vecs_array = np.array(self.class_vecs)
      num_class_frames = len(class_vecs_array)

      # Compute mean vectors using array slicing (faster than list comprehension)
      num_segments = (num_class_frames + class_smooth_frames - 1) // class_smooth_frames
      class_frames_interp = []
      for i in range(0, num_class_frames, class_smooth_frames):
        end_idx = min(i + class_smooth_frames, num_class_frames)
        class_frames_interp.append(np.mean(class_vecs_array[i:end_idx], axis=0))

      # Interpolate
      self.class_vecs = full_frame_interpolation(class_frames_interp,
                                            class_smooth_frames,
                                            num_class_frames)
      

  def setup_effects(self):
    '''initialises effects to be applied to each frame'''

    self.custom_effects = self.custom_effects or []
    start = self.start
    duration = self.duration

    # initialise pre-made Contrast effect 
    if all(var is None for var in [self.contrast_audio, 
                                  self.contrast_strength,
                                  self.contrast_percussive]):
      pass
    else:
      self.contrast_audio = self.contrast_audio or self.song
      self.contrast_strength = self.contrast_strength or 0.5
      self.contrast_percussive = self.contrast_percussive or True

      contrast = EffectsGenerator(audio = self.contrast_audio, 
                                  func = contrast_effect,
                                  strength = self.contrast_strength, 
                                  percussive = self.contrast_percussive)
      self.custom_effects.append(contrast)

    # initialise pre-made Flash effect
    if all(var is None for var in [self.flash_audio, 
                                  self.flash_strength,
                                  self.flash_percussive]):
      pass
    else:
      self.flash_audio = self.flash_audio or self.song
      self.flash_strength = self.flash_strength or 0.5
      self.flash_percussive = self.flash_percussive or True
  
      flash = EffectsGenerator(audio = self.flash_audio, 
                                  func = flash_effect, 
                                  strength = self.flash_strength, 
                                  percussive = self.flash_percussive)
      self.custom_effects.append(flash)

    # initialise Custom effects
    for effect in self.custom_effects:
      effect.audio = effect.audio or self.song
      effect.render_audio(start=start, 
                          duration = duration, 
                          n_mels = self.input_shape, 
                          hop_length = self.frame_duration)

  def generate_frames(self):
    '''Generate GAN output for each frame of video'''

    file_name = self.file_name
    resolution = self.resolution
    batch_size = self.batch_size
    frame_batch_size = self.frame_batch_size
    
    num_frame_batches = int(len(self.noise) / batch_size)
    max_frame_index = num_frame_batches * batch_size + batch_size

    on_disk = not(bool(frame_batch_size != None) & bool(max_frame_index <= frame_batch_size))

    if self.use_tf:
        Gs_syn_kwargs = {'output_transform': {'func': self.convert_images_to_uint8, 
                                          'nchw_to_nhwc': True},
                    'randomize_noise': False,
                    'minibatch_size': batch_size}
    else:
        Gs_syn_kwargs = {'noise_mode': 'const'} # random, const, None

    # Set-up temporary frame directory
    # Fixme: Save images to RAM
    self.frames_dir = str(Path(file_name).stem) + '_frames'
    
    if on_disk:
      if os.path.exists(self.frames_dir):
          shutil.rmtree(self.frames_dir)
      os.makedirs(self.frames_dir)

#TODO: Create "model resolution", maybe I can detect this directly from the model?? That way we only resize if the model resolution is different

    if on_disk:
      all_frames = np.empty(shape=[frame_batch_size, resolution, resolution, 3], dtype=np.uint8)
    else:
      all_frames = np.empty(shape=[num_frame_batches, resolution, resolution, 3], dtype=np.uint8)

    # Use stored device from model loading, or create new one for custom styles
    device = getattr(self, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate frames
    num_batches = 0
    frame_count = 0
    for i in tqdm(range(num_frame_batches), position=0, leave=True, desc="Generating frames"):
        if on_disk and frame_count == frame_batch_size:
            # if batch size met, frames write to disk, reset array 
            # Save. Include leading zeros in file name to keep alphabetical order
          for f in tqdm(range(frame_count), position=0, leave=True):
            file_name = str(num_batches*frame_batch_size + f)\
                    .zfill(len(str(max_frame_index)))
            Image.fromarray(all_frames[f], 'RGB').save(os.path.join(self.frames_dir, file_name + '.jpg'), quality=95) #, subsample=0, quality=95)

          num_batches += 1
          frame_count = 0
          # Don't need to actually do this, we just use the same array and overwrite it
          # all_frames = np.empty(shape=[frame_batch_size, resolution, resolution, 3], dtype=np.uint8)

        # Obtain batches of Noise and Class vectors based on batch_size
        noise_batch = np.array(self.noise[i*batch_size:(i+1)*batch_size], dtype=np.float32)
        class_batch = np.array(self.class_vecs[i*batch_size:(i+1)*batch_size], dtype=np.float32)

        # If style is a custom function, pass batches to the function
        if callable(self.style):
            image_batch = self.style(noise_batch=noise_batch,
                                   class_batch=class_batch)
        # Otherwise, generate frames with StyleGAN
        else:
            if self.use_tf:
                w_batch = self.Gs.components.mapping.run(noise_batch, np.tile(class_batch, (batch_size, 1)))
                image_batch = self.Gs.components.synthesis.run(w_batch, **Gs_syn_kwargs)
            else:
                # Convert to tensor and move to device
                noise_tensor = torch.from_numpy(noise_batch).to(device)

                # Use mixed precision if enabled
                if self.use_fp16:
                    if self.use_r3gan and hasattr(torch, 'bfloat16'):
                        # R3GAN prefers BFloat16 over FP16
                        noise_tensor = noise_tensor.to(torch.bfloat16)
                    else:
                        noise_tensor = noise_tensor.half()

                # Use inference_mode for best performance (faster than no_grad)
                with torch.inference_mode():
                    # R3GAN and StyleGAN both support mapping + synthesis interface
                    # R3GAN also supports direct G(z, c) call, but mapping+synthesis
                    # gives us access to W-space for smoother interpolation
                    try:
                        w_batch = self.Gs.mapping(noise_tensor, class_batch)
                        image_batch = self.Gs.synthesis(w_batch, **Gs_syn_kwargs)
                    except (AttributeError, TypeError):
                        # Fallback to direct call if mapping/synthesis not available
                        # This handles some R3GAN model variations
                        class_tensor = torch.from_numpy(class_batch).to(device)
                        if self.use_fp16:
                            if self.use_r3gan and hasattr(torch, 'bfloat16'):
                                class_tensor = class_tensor.to(torch.bfloat16)
                            else:
                                class_tensor = class_tensor.half()
                        image_batch = self.Gs(noise_tensor, class_tensor)

                # Move to CPU for post-processing
                image_batch = image_batch.detach().cpu().float()

        # For each image in generated batch: apply effects, resize, and save
        for j, image in enumerate(image_batch):
            image_index = (i * batch_size) + j
            if not self.use_tf:
                image = (image.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
            array = np.array(image)

            # Apply effects
            for effect in self.custom_effects:
                array = effect.apply_effect(array = array, 
                                            index = image_index)

            # final_image = Image.fromarray(array, 'RGB')

            # If resolution is provided, resize
            #if resolution:
            #    final_image = final_image.resize((resolution, resolution))

            all_frames[frame_count] = array
            frame_count += 1
        
        del image_batch
        del noise_batch
    
    # write remaining frames
    if on_disk:
      for f in tqdm(range(frame_count), position=0, leave=True):
        file_name = str(num_batches*frame_batch_size + f)\
                .zfill(len(str(max_frame_index)))
        Image.fromarray(all_frames[f], 'RGB').save(os.path.join(self.frames_dir, file_name + '.jpg'), quality=95) #, subsample=0, quality=95)

      return None
    
    return all_frames


  def hallucinate(self,
                  file_name: str, 
                  output_audio: str = None,
                  fps: int = 30, 
                  resolution: int = None, 
                  start: float = 0, 
                  duration: float = None, 
                  save_frames: bool = False,
                  batch_size: int = 1,
                  frame_batch_size: int = None,
                  speed_fpm: int = 12,
                  pulse_percussive: bool = True,
                  pulse_harmonic: bool = False,
                  pulse_react: float = 0.5,
                  motion_percussive: bool = False,
                  motion_harmonic: bool = True,
                  motion_react: float = 0.5, 
                  motion_randomness: float = 0.5,
                  truncation: float = 1,
                  classes: list = None,
                  dominant_classes_first: bool = False,
                  class_pitch_react: float = 0.5,
                  class_smooth_seconds: int = 1,
                  class_complexity: float = 1, 
                  class_shuffle_seconds: float = None,
                  class_shuffle_strength: float = 0.5,
                  contrast_strength: float = None, 
                  contrast_percussive: bool = None,
                  flash_strength: float = None,
                  flash_percussive: bool = None,
                  custom_effects: list = None):
    '''Full pipeline of video generation'''

    # Raise exception if speed_fpm > fps*60
    if speed_fpm > fps*60:
      raise InvalidParameterError('speed_fpm must not be greater than fps * 60')

    # Raise exception if element of custom_effects is not EffectsGenerator
    if custom_effects:
      if not all(isinstance(effect, EffectsGenerator) \
                  for effect in custom_effects):
        raise InvalidParameterError('Elements of custom_effects must be EffectsGenerator objects')

    # Raise exception of classes is an empty list
    if classes:
      if len(classes) == 0:
        raise InvalidParameterError('classes must be NoneType or list with length > 0')

    # Raise exception if any of the following parameters are not between 0 and 1
    for param in ['motion_randomness', 'truncation','class_shuffle_strength',
                  'contrast_strength', 'flash_strength']:

        if (locals()[param]) and not (0 <= locals()[param] <= 1):
          raise InvalidParameterError(f'{param} must be between 0 and 1')

    # Warn about memory usage on Colab
    if resolution and is_colab():
        estimated_mem = estimate_memory_usage_mb(resolution, batch_size)
        available_mem = get_gpu_memory_mb()
        if available_mem > 0 and estimated_mem > available_mem * 0.8:
            suggested = suggest_batch_size(resolution, available_mem)
            print(f"Warning: Estimated memory usage ({estimated_mem:.0f}MB) may exceed "
                  f"available GPU memory ({available_mem:.0f}MB). "
                  f"Consider using batch_size={suggested}")

    # Use pathlib for robust file name handling
    file_path = Path(file_name)
    self.file_name = str(file_path.with_suffix('.mp4'))
    self.resolution = resolution
    self.batch_size = batch_size
    self.frame_batch_size = frame_batch_size
    self.speed_fpm = speed_fpm
    self.pulse_react = pulse_react
    self.motion_react = motion_react 
    self.motion_randomness = motion_randomness
    self.truncation = truncation
    self.classes = classes
    self.dominant_classes_first = dominant_classes_first
    self.class_pitch_react = class_pitch_react
    self.class_smooth_seconds = class_smooth_seconds
    self.class_complexity = class_complexity
    self.class_shuffle_seconds = class_shuffle_seconds
    self.class_shuffle_strength = class_shuffle_strength
    self.contrast_strength = contrast_strength
    self.contrast_percussive = contrast_percussive
    self.flash_strength = flash_strength
    self.flash_percussive = flash_percussive
    self.custom_effects = custom_effects 

    # initialise style
    if not self.style_exists:

      print('Preparing style...')

      if not callable(self.style):
        self.stylegan_init()

      self.style_exists = True

    # If there are changes in any of the following parameters,
    # re-initialise audio
    cond_list = [(not hasattr(self, 'fps')) or (self.fps != fps),
                 (not hasattr(self, 'start')) or (self.start != start),
                 (not hasattr(self, 'duration')) or (self.duration != duration),
                 (not hasattr(self, 'pulse_percussive')) or \
                 (self.pulse_percussive != pulse_percussive),
                 (not hasattr(self, 'pulse_harmonic')) or \
                 (self.pulse_percussive != pulse_harmonic),
                 (not hasattr(self, 'motion_percussive')) or \
                 (self.motion_percussive != motion_percussive),
                 (not hasattr(self, 'motion_harmonic')) or \
                 (self.motion_percussive != motion_harmonic)]

    if any(cond_list):
      
      self.fps = fps
      self.start = start
      self.duration = duration 
      self.pulse_percussive = pulse_percussive
      self.pulse_harmonic = pulse_harmonic
      self.motion_percussive = motion_percussive
      self.motion_harmonic = motion_harmonic

      print('Preparing audio...')
      self.load_specs()

    # initialise effects
    print('Loading effects...')
    self.setup_effects()
    
    # Transform/assign value of classes
    self.transform_classes()

    # Generate vectors
    print('\nDoing math...\n')
    self.generate_vectors()

    # Generate frames
    print('\nHallucinating... \n')
    all_frames = self.generate_frames()

    # Load output audio
    if output_audio:
        wav_output, sr_output = librosa.load(output_audio, offset=start, duration=duration)
    else:
        wav_output, sr_output = self.wav, self.sr

    # Use temp directory for temporary files (Colab-compatible)
    temp_dir = get_temp_dir()
    temp_video_path = os.path.join(temp_dir, 'lsd_tmp.mp4')
    temp_audio_path = os.path.join(temp_dir, 'lsd_tmp.wav')

    # Write temporary movie file
    print('\nGenerating movie...\n')
    if np.any(all_frames):  # If frames not passed back, then they're in folder
        imageio.mimwrite(temp_video_path, all_frames, quality=8, fps=self.sr/self.frame_duration)
        video = VideoFileClip(temp_video_path)
    else:
        video = ImageSequenceClip(self.frames_dir, fps=self.sr/self.frame_duration)

    # Write temporary audio file - fixing the soundfile write
    try:
        # Ensure audio data is normalized float32 between -1 and 1
        wav_output = np.array(wav_output, dtype=np.float32)
        if wav_output.max() > 1.0 or wav_output.min() < -1.0:
            wav_output = wav_output / max(abs(wav_output.max()), abs(wav_output.min()))

        soundfile.write(temp_audio_path, wav_output, sr_output, format='WAV')
    except Exception as e:
        print(f"Error writing audio file: {e}")
        raise

    # Mix audio & video with codec fallback for Colab compatibility
    try:
        audio = AudioFileClip(temp_audio_path, fps=self.sr*2)
        video = video.set_audio(audio)
        write_video_with_codec_fallback(video, self.file_name, audio_bitrate="1024k")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if np.any(all_frames) and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if not save_frames and not np.any(all_frames) and os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        del all_frames

    print(f'\nVideo saved to: {self.file_name}\n')


class EffectsGenerator:
  def __init__(self, 
               func, 
               audio: str = None,
               strength: float = 0.5,
               percussive: bool = True):
    self.audio = audio
    self.func = func 
    self.strength = strength
    self.percussive = percussive

    # Raise exception if func does not take in parameters array,
    # strength, and amplitude
    func_sig = list(inspect.getfullargspec(func))[0]
    for arg in ['array', 'strength', 'amplitude']:
      if arg not in func_sig:
        raise InvalidParameterError(
            'func must be a function with parameters array, strength, and amplitude'
        )
    

  def render_audio(self, start, duration, n_mels, hop_length):
    '''Prepare normalized spectrogram of audio to be used for effect'''
    
    # Now with dirty hax for rounded fps values!
    # Previous max ~40fps, this is for 60fps
    sample_rate = 30720*2

    # Load spectrogram 
    wav, sr = librosa.load(self.audio, sr=sample_rate, offset=start, duration=duration)

    # If percussive = True, decompose harmonic and percussive signals
    if self.percussive: 
      wav = librosa.effects.hpss(wav)[1]

    # Get normalized spectrogram  
    self.spec = get_spec_norm(wav, sr, n_mels=n_mels, hop_length=hop_length)


  def apply_effect(self, array, index):
    '''Apply effect to image (array)'''

    amplitude = self.spec[index]
    return self.func(array=array, strength = self.strength, amplitude=amplitude)
