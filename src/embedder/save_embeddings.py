import embedding_model as model
import embedder_utils as ut
import glob
import numpy as np
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
        'vlad_cluster':8,
        'batch_size':16,
        'bottleneck_dim':512,
        'aggregation_mode':'gvlad',
        'model':'../../models/embedder.h5',
        'loss':'softmax'}
network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                            num_class=params['n_classes'],
                                            mode='eval', args=args)
network_eval.load_weights(args['model'], by_name=True)


for speaker in glob.glob("/mnt/md0/datasets/VCTK-Corpus/wav48/*"):
    path = glob.glob(speaker + "/*")[2] #third sentence is longer
    specs = ut.load_data(path, win_length=params['win_length'], sr=params['sampling_rate'],
                     hop_length=params['hop_length'], n_fft=params['nfft'],
                     spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    v = network_eval.predict(specs)
    np.save("../embeddings/" + speaker[-4:] + ".npy", v)
