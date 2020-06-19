import librosa
import numpy as np
import tensorflow as tf
import os
import warnings
import sys
from python_speech_features import get_filterbanks
from scipy.fft import fft

from audio import training_generator
from convert_voice import convert_voice
from constants import NUM_FRAMES
from constants import EMB_FRAMES
from constants import NUM_FBANKS
from constants import DATA_DIR
from constants import SAMPLE_RATE
from constants import BATCH_SIZE
from constants import MODEL
from constants import NFFT
from constants import PREEMPH
from constants import ALPHA
from constants import BETA
from constants import learning_rate

from model import ACVAE
import matplotlib.pyplot as plt

tf.enable_eager_execution()  

def loss_func(model, x, f0, emb):
    """
    x - normalized log mel frequency energies: batch_size x NUM_FBANKS x NUM_FRAMES tensor
    f0 - pitch
    emb - embedding
    """
    x = tf.expand_dims(x, axis=-1)
#     norm_f0 = tf.math.log(f0 + 1)
#     norm_f0 = (norm_f0 - tf.math.reduce_mean(norm_f0)) / tf.math.reduce_std(norm_f0)
#     norm_f0 = tf.zeros_like(norm_f0)
#    emb = tf.zeros_like(emb)
    recon_x, mu, logvar = model([x, emb, emb, f0])
    
    assert(x.shape == recon_x.shape), (x.shape, recon_x.shape)
    L1 = tf.math.reduce_mean(tf.abs(recon_x - x))
    
    #Kullback-Leibler divergence     
    KLD = -0.5 * tf.math.reduce_mean(1 + logvar - mu**2 - tf.exp(logvar))
            
    return L1, KLD, L1 + ALPHA * KLD


def grad(model, x, f0, emb):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        L1, KLD, loss_value = loss_func(model, x, f0, emb)
    return L1, KLD, loss_value, tape.gradient(loss_value, model.trainable_variables)


def main():
    if not os.path.exists("../converted_voices"):
        os.mkdir("../converted_voices")
    if not os.path.exists("../figure"):
        os.mkdir("../figure")
        
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    train_dataset = tf.data.Dataset.from_generator(training_generator,
                                             output_types=(np.float32, np.float32, np.float32)).batch(BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()
    
    acvae = ACVAE()
    model = acvae.model
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model, iterator=train_dataset)
    manager = tf.train.CheckpointManager(ckpt, MODEL, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    print("Learning rate", optimizer.learning_rate.numpy())
    optimizer.learning_rate.assign(learning_rate)
    
    
    wav1 = "/mnt/md0/datasets/VCTK-Corpus/wav48/p262/p262_027.wav" #woman source
    wav2 = "/mnt/md0/datasets/VCTK-Corpus/wav48/p256/p256_002.wav" #man target
    emb1 = np.load("../embeddings/p262.npy")
    emb2 = np.load("../embeddings/p256.npy")
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    L1_loss = tf.keras.metrics.Mean("L1_loss", dtype=tf.float32)
    KLD_loss = tf.keras.metrics.Mean("KLD", dtype=tf.float32)
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    
    num_epochs = 10000
    for epoch in range(num_epochs):
        for tup in train_dataset:
            x, emb, f0 = tup
            L1, KLD, loss, grads = grad(model, x, f0, emb)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss(loss)
            L1_loss(L1)
            KLD_loss(KLD)
            ckpt.step.assign_add(1)
            if int(ckpt.step) % 30 == 0:
                save_path = manager.save(),
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            

        wav_converted = convert_voice(acvae, wav1, wav2, emb1, emb2)
        if epoch % 10 == 0:
            librosa.output.write_wav("../converted_voices/conv_{}.wav".format(epoch),
                                 wav_converted,
                                 sr=SAMPLE_RATE,
                                 norm=True)

        template = 'Epoch {}, TrainLoss: {:.3}, KLD_loss: {:.2}, L1_Loss: {:.2}'
        print (template.format(epoch+1,
                              train_loss.result(), 
                              KLD_loss.result(),
                              L1_loss.result()
                              ))
        
        # Reset metrics every epoch
        train_loss.reset_states()
        L1_loss.reset_states()
        val_loss.reset_states()
        KLD_loss.reset_states()
#model.save("models/cvae_{}.tf".format(epoch), save_format="tf") 
if __name__ == "__main__":
    main()


