import numpy as np
import random
import os
from scipy.fft import fft
from scipy.signal import lfilter
from python_speech_features import get_filterbanks
import librosa
import pyworld
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import glob

from constants import NFFT
from constants import SAMPLE_RATE
from constants import FRAME_PERIOD
from constants import DATA_DIR
from constants import NUM_FBANKS
from constants import NUM_FRAMES
from constants import EMB_FRAMES
from constants import PREEMPH

fb = get_filterbanks(NUM_FBANKS,NFFT,SAMPLE_RATE,lowfreq=0,highfreq=None)
filter_sums = fb.sum(axis=-1, keepdims=True).T
filter_centers = fb.argmax(axis=-1)
filter_centers = np.array([filter_centers[0] - 1] + list(filter_centers) + [filter_centers[-1] + 1])

def _histc(x, edges, index):
    """
    assume x is sorted in ascending order
    index[i] = number of elements in x smaller or equal than edges[i],
    truncated to (1, len(x) - 1) (0 replaced by 1, len(x) by len(x) - 1)
    """
    x_length = len(x)
    edges_length = len(edges)
    count = 1
    i = 0
    while i < edges_length and edges[i] < x[0]:
        index[i] = 1
        i += 1
    while i < edges_length and count < x_length:
        if edges[i] < x[count]:
            index[i] = count
            i += 1
        else:
            index[i] = count
            count += 1
    count -= 1
    i += 1
    while i < edges_length:
        index[i] = count
        i += 1
    return

def _interp1(x, y, xi):
    """
    Given a function y = y(x),
    return its values in points xi
    (linear interpolation)
    x must be an increasing sequence of floats
    """
    x_length = len(x)
    xi_length = len(xi)
    yi = np.zeros(xi_length)
    k = np.zeros(xi_length, dtype=int)
    h = x[1:] - x[:-1]
  
    _histc(x, xi, k)

    for i in range(xi_length):
        s = (xi[i] - x[k[i] - 1]) / h[k[i] - 1]
        yi[i] = y[k[i] - 1] + s * (y[k[i]] - y[k[i] - 1])
    return yi

def mfe2sp(mfe):
    """
    mfe - mel-frequency energies, NUM_FRAMES x NUM_FBANKS
    Returns power spectrum, NUM_FRAMES x NFFT
    """
    assert mfe.shape[1] == NUM_FBANKS
    arr = mfe / filter_sums
    sp = []
    for line in arr:
        line = [line[0]] + list(line) + [line[-1]]
        sp.append(_interp1(filter_centers, line, np.arange(NFFT // 2 + 1)))
    return np.maximum(sp, 1e-16)
  
def sp2mfe(sp):
    """
    sp - power spectrum, NUM_FRAMES x NFFT
    Returns: mel-frequency energies, NUM_FRAMES x NUM_FBANKS
    """
    return np.dot(sp,fb.T)
  
def get_embedding(filename):
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }
    args = {'gpu':1,
            'net':'resnet34s',
            'ghost_cluster':2,
            'vlad_cluster':8}

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)
def get_features(filename, training=True):
    """
    For training, we only need mel frequency energies
    For conversion, also power spectrum etc.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wav, _ = librosa.load(filename, 
                              sr=SAMPLE_RATE, 
                              mono=True,
                              dtype=np.float64)
    energy = np.abs(wav)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    if training:
        audio_voice_only = wav[offsets[0]:offsets[-1]]
    else:
    #avoid cutting off too abruptly
        audio_voice_only = wav[offsets[0]:offsets[-1] + 4800]
    if training:
        if len(audio_voice_only) >= 160 * NUM_FRAMES:
            start_ = np.random.randint(len(audio_voice_only) - 160 * NUM_FRAMES + 1)
            end_ = start_ + 160 * NUM_FRAMES - 1
            audio_voice_only = audio_voice_only[start_:end_]
        else:
            return [0], [0]
    wav = librosa.util.normalize(audio_voice_only)
    #deep speaker uses preemphasis here, I do not, because I want the model to correctly transform lower
    #frequencies, too. I apply preemphasis to spectrum before putting data into model embedder instead.
    wav = lfilter([1., -PREEMPH], [1.], wav)[1:]
    #f0 extraction (most time consuming operation in this function)
    f0, timeaxis = pyworld.harvest(wav, SAMPLE_RATE, frame_period=FRAME_PERIOD, f0_floor=71.0, f0_ceil=800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, SAMPLE_RATE, fft_size=NFFT)
    ap = pyworld.d4c(wav, f0, timeaxis, SAMPLE_RATE, fft_size=NFFT)
    mfe = sp2mfe(sp)
    lmfe = np.log(mfe)
    mean = np.mean(lmfe)
    std = np.std(lmfe)
    nmfe = (lmfe - mean) / std
    
    if training:
        return nmfe.T, f0
    else:
        out_len = len(f0) // 4 * 4
#        out_len = len(f0)
        return nmfe[:out_len].T, mean, std, sp[:out_len], f0[:out_len], ap[:out_len]

# use VCTK dataset
test_speakers = ['p225', 'p226', 'p270', 'p233', 'p262','p256']

DATA_DIR = "/mnt/md0/datasets/VCTK-Corpus/wav48/"
speakers = os.listdir(DATA_DIR)
for st in test_speakers:
    speakers.remove(st)
    
def quantize(f):
    f = np.array(f)
    res = np.zeros((len(f), 257))
    res[np.where(f==0), 256] = 1
    if np.all(f==0):
        mean = 0
        std = 1
    else:
        mean = np.mean(f[np.where(f>0)])
        std = np.std(f[np.where(f>0)]) + 1e-16
    f = (f - mean) / std / 4
    for i in range(256):
        res[(f > (2*i -1) / 256) & (f <= (2*i + 1) / 256), i] = 1
    return res

#speaker_num = {speaker:num for num, speaker in enumerate(speakers)}
def training_generator():
    for _ in tqdm(range(40)):
        speaker = np.random.choice(speakers)
        emb = np.load("../embeddings/" + speaker + ".npy")
        files = glob.glob(DATA_DIR + "/" + speaker + "/*")
        file = np.random.choice(files)
        if file[-4:] == ".wav":
            nmfe, f0 = get_features(file)
            f0 = quantize(f0)
            if len(nmfe) > 1:
                if nmfe.shape[1] == NUM_FRAMES:
                    yield (nmfe.astype(np.float32), emb.astype(np.float32), np.expand_dims(f0.astype(np.float32),
                          axis=0))#, speaker_num[speaker])

            
def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0 + 1e-16) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted
    