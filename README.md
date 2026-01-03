# Lucid Sonic Dreams
Lucid Sonic Dreams syncs GAN-generated visuals to music!

Supports multiple GAN architectures:
- **R3GAN** (NEW!) - Modern GAN baseline with state-of-the-art FID scores ([Paper](https://arxiv.org/abs/2501.05441))
- **StyleGAN3** - NVIDIA's alias-free GAN ([Repo](https://github.com/NVlabs/stylegan3))
- **StyleGAN2** - Pre-trained models from [Justin Pinkney's repository](https://github.com/justinpinkney/awesome-pretrained-stylegan2)

Sample output can be found on [YouTube](https://www.youtube.com/watch?v=SDf7a28cSVs).

## Installation  
  
This implementation has been tested on Python 3.9. It now uses the PyTorch implementation of StyleGAN2-ada, and works with Ampere cards.

To install:
git clone this repo and change directory into your newly created directory:

```
git clone https://github.com/nerdyrodent/lucid-sonic-dreams.git
cd lucid-sonic-dreams
```

It is suggested that Anaconda or Miniconda be used to create a new, virtual Python environment with a name of your choice. For example:

```
conda create --name sonicstylegan-pt python=3.9
conda activate sonicstylegan-pt
```

Install the packages required for both stylegan2-ada-pytorch and this repo:

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install click requests ninja imageio imageio-ffmpeg tqdm psutil scipy pyspng
```

(Optional) If you already have stylegan2-ada-pytorch (recommended for training your own networks), create a symbolic link to it:

`ln -s ../stylegan2-ada-pytorch stylegan2`


## Usage

Refer to the [Lucid Sonic Dreams Tutorial Notebook](https://colab.research.google.com/drive/1Y5i50xSFIuN3V4Md8TB30_GOAtts7RQD?usp=sharing) for full parameter descriptions and sample code templates. A basic visualization snippet is also found below.

### Basic Visualization

```python
from lucidsonicdreams import LucidSonicDream

L = LucidSonicDream(song = 'song.mp3',
                    style = 'abstract photos')

L.hallucinate(file_name = 'song.mp4')
```

### R3GAN Models (Recommended)

R3GAN achieves better visual quality (lower FID) than StyleGAN2 with a simpler architecture:

```python
from lucidsonicdreams import LucidSonicDream, show_r3gan_styles

# See available R3GAN models
show_r3gan_styles()

# Use R3GAN for face generation (FID: 2.75 vs StyleGAN2's 3.78)
L = LucidSonicDream(song='song.mp3', style='r3gan_ffhq_256')
L.hallucinate(file_name='faces.mp4')

# Use R3GAN for class-conditional generation (1000 ImageNet classes)
L = LucidSonicDream(song='song.mp3', style='r3gan_imagenet_64')
L.hallucinate(file_name='objects.mp4')
```

**Available R3GAN Models:**
| Model | Resolution | Classes | FID | Description |
|-------|------------|---------|-----|-------------|
| `r3gan_ffhq_256` | 256x256 | 0 | 2.75 | High-quality faces |
| `r3gan_ffhq_64` | 64x64 | 0 | 1.95 | Fast face generation |
| `r3gan_cifar10` | 32x32 | 10 | 1.96 | 10-class conditional |
| `r3gan_imagenet_64` | 64x64 | 1000 | 2.09 | 1000-class conditional |
| `r3gan_imagenet_32` | 32x32 | 1000 | 1.27 | 1000-class conditional |

sg2-ada-pt-song-spleeter.py is an example with a variety of configuration options as defaults, based on the audio being split into 4 stems using spleeter - https://github.com/deezer/spleeter

