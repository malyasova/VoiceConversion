import numpy as np
import librosa
from model import ACVAE
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import pyworld
import os
from scipy.fft import fft
from python_speech_features import get_filterbanks
from scipy.signal import lfilter

tf.enable_eager_execution()
from audio import get_features
from audio import pitch_conversion
from audio import mfe2sp
from constants import MODEL
from constants import SAMPLE_RATE
from constants import EMB_FRAMES
from constants import NUM_FBANKS
from constants import embedder_model
from constants import PREEMPH
from constants import FRAME_PERIOD
from constants import NFFT
import embedder.embedding_model as embedding_model
import embedder.embedder_utils as ut
from audio import quantize

fb = get_filterbanks(NUM_FBANKS,NFFT,SAMPLE_RATE,lowfreq=0,highfreq=None)
filter_centers = fb.argmax(axis=-1)
preemph_transform = np.abs(fft([1., -PREEMPH] + [0] * (NFFT - 2)))[:NFFT //2 + 1]**2

def convert_voice(model, wav_s, wav_t, emb_s, emb_t):
    """Arguments:
    cvae - ACVAE model
    embedder - DeepSpeakerModel
    wav_s - source voice
    wav_t - target voice
    Returns: 
    wav file with words from source voice, voice from target voice
    """
    pic_dir = "../figure/"
    if not os.path.exists(pic_dir):
        os.mkdir(pic_dir)
    feat_t, mean_t, std_t, sp_t, f0_t, ap_t = get_features(wav_t, training=False)
    feat_s, mean_s, std_s, sp_s, f0_s, ap_s = get_features(wav_s, training=False)
    
    logf0s_mean_s = np.mean(np.ma.log(f0_s))
    logf0s_std_s = np.std(np.ma.log(f0_s))
    logf0s_mean_t = np.mean(np.ma.log(f0_t))
    logf0s_std_t = np.std(np.ma.log(f0_t))
    
    f0_converted = pitch_conversion(f0=f0_s, 
                                    mean_log_src=logf0s_mean_s, 
                                    std_log_src=logf0s_std_s, 
                                    mean_log_target=logf0s_mean_t, 
                                    std_log_target=logf0s_std_t)

    mu_enc, logvar_enc = model.encoder([np.expand_dims(feat_s.astype(np.float32), [0,-1]), emb_s.astype(np.float32)])
    z_enc = model.reparameterize(mu_enc, logvar_enc)
#    norm_f0_c = quantize([1]*len(f0_s))
    norm_f0 = quantize(f0_s)
    
    nmfe_converted = model.decoder([z_enc, tf.reshape(emb_t,(1,-1)), np.expand_dims(norm_f0.astype(np.float32), [0,1])])
    nmfe_recon = model.decoder([z_enc, tf.reshape(emb_s,(1,-1)), np.expand_dims(norm_f0.astype(np.float32), [0,1])])
    nmfe_converted = np.squeeze(nmfe_converted.numpy())

#    print("Val L1 loss: {}".format(np.mean(np.abs(np.squeeze(nmfe_recon.numpy()) - feat_s)).mean()))
    nmfe_recon = np.squeeze(nmfe_recon.numpy())
        
    mfe_converted = np.exp(nmfe_converted.T * std_t + mean_t)
    mfe_recon = np.exp(nmfe_recon.T * std_s + mean_s)

    plt.imshow(nmfe_converted[:100])
    plt.colorbar()
    plt.savefig(pic_dir + "convert.png")
    plt.close("all")
    plt.imshow(nmfe_recon[:100])
    plt.colorbar()
    plt.savefig(pic_dir + "recon.png")
    plt.close("all")
    plt.imshow(feat_s[:100])
    plt.colorbar()
    plt.savefig(pic_dir + "orig.png")
    plt.close("all")

    sp_converted_s = mfe2sp(mfe_recon)
    sp_converted_t = mfe2sp(mfe_converted)

    plt.imshow(sp_converted_s)
    plt.colorbar()
    plt.savefig(pic_dir + "sp_recovered.png")
    plt.close("all")
    plt.imshow(sp_converted_t)
    plt.colorbar()
    plt.savefig(pic_dir + "sp_converted.png")
    plt.close("all")
    factor = np.divide(sp_converted_t, sp_converted_s)
        
    sp_gained = np.multiply(sp_s[:len(factor)], factor[:len(sp_s)])
    #remove too big peaks:
#    sp_gained = np.minimum(sp_gained, sp_s.max(axis=1, keepdims=True)[:len(sp_gained)])

    plt.plot(mfe_recon[50], color="green")
    plt.plot(mfe_converted[50], color="red")
    plt.plot(feat_s[50], color="black")
    plt.savefig(pic_dir + "mfe.png")
    plt.close("all")
    plt.figure(figsize=(5,10))
    plt.plot(sp_converted_t[50], color="red")
    plt.plot(sp_converted_s[50], color="green")
    plt.plot(sp_s[50], color="black")
    plt.plot(sp_gained[50], color="blue")
    plt.savefig(pic_dir + "sp.png")
    plt.close("all")
    plt.plot(sp_converted_t[10], color="red")
    plt.plot(sp_converted_s[10], color="green")
    plt.plot(sp_s[10], color="black")
    plt.plot(sp_gained[10], color="blue")
    plt.savefig(pic_dir + "sp2.png")
    plt.close("all")
    plt.imshow(sp_t)
    plt.savefig(pic_dir + "target_sp.png")
    plt.imshow(sp_s)
    plt.savefig(pic_dir + "sp_orig.png")
    plt.close("all")
    #deemphasis
    plt.imshow(sp_gained)
    plt.savefig(pic_dir + "sp_gained.png")

    sp_gained = sp_gained / preemph_transform.reshape(1,-1)
    
    wav_transformed = pyworld.synthesize(f0_converted[:len(sp_gained)], 
#    wav_transformed = pyworld.synthesize(np.array([np.exp(logf0s_mean_t)] * len(sp_gained)),
                                         sp_gained, 
                                         ap_s[:len(sp_gained)], 
                                         SAMPLE_RATE,
                                         FRAME_PERIOD)
    #normalize amplitude
    wav_result = librosa.util.normalize(wav_transformed)
    wav_result = wav_result.astype(np.float32)
    return wav_result

def main():
    path1, path2, write_path = sys.argv[1], sys.argv[2], sys.argv[3]
#TO DO: resolve placeholder / eager execution conflict somehow to make commented code work
#     params = {'dim': (257, None, 1),
#           'nfft': 512,
#           'spec_len': 250,
#           'win_length': 400,
#           'hop_length': 160,
#           'n_classes': 5994,
#           'sampling_rate': 16000,
#           'normalize': True,
#           }
#     args = {'gpu':1,
#             'net':'resnet34s',
#             'ghost_cluster':2,
#             'vlad_cluster':8,
#             'batch_size':16,
#             'bottleneck_dim':512,
#             'aggregation_mode':'gvlad',
#             'model':'../models/embedder.h5',
#             'loss':'softmax'}
#     embedder = embedding_model.vggvox_resnet2d_icassp(input_dim=params['dim'],
#                                                 num_class=params['n_classes'],
#                                                 mode='eval', args=args)
#     embedder.load_weights(args['model'], by_name=True)
#     specs1 = ut.load_data(path1, win_length=params['win_length'], sr=params['sampling_rate'],
#                      hop_length=params['hop_length'], n_fft=params['nfft'],
#                      spec_len=params['spec_len'], mode='eval')
#     specs1 = np.expand_dims(np.expand_dims(specs1, 0), -1)
#     emb1 = embedder.predict(specs1)
#     specs2 = ut.load_data(path2, win_length=params['win_length'], sr=params['sampling_rate'],
#                      hop_length=params['hop_length'], n_fft=params['nfft'],
#                      spec_len=params['spec_len'], mode='eval')
#     specs2 = np.expand_dims(np.expand_dims(specs2, 0), -1)
#     emb2 = embedder.predict(specs2)
    speaker1 = path1[-12:-8]
    speaker2 = path2[-12:-8]
    emb1 = np.load("../embeddings/" + speaker1 + ".npy")
    emb2 = np.load("../embeddings/" + speaker2 + ".npy")

    acvae = ACVAE()
    model = acvae.model
    ckpt = tf.train.Checkpoint(net=model)
    manager = tf.train.CheckpointManager(ckpt, MODEL, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint).expect_partial()

    result = convert_voice(acvae, path1, path2, emb1, emb2)
    librosa.output.write_wav(write_path,
                             result[:40000],
                             sr=SAMPLE_RATE,
                             norm=True)

if __name__ == "__main__":
    main()