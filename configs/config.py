import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    seq_len = 20 # input timesteps

    root_dir = '/mnt/qb/geiger/kchitta31/data_06_21'
    train_data, val_data = [], []
    for town in train_towns:
        train_data.append(os.path.join(root_dir, town+'_tiny'))
        train_data.append(os.path.join(root_dir, town+'_short'))
    for town in val_towns:
        val_data.append(os.path.join(root_dir, town+'_short'))

    # visualizing transformer attention maps
    viz_root = '/mnt/qb/geiger/kchitta31/data_06_21'
    viz_towns = ['Town05_tiny']
    viz_data = []
    for town in viz_towns:
        viz_data.append(os.path.join(viz_root, town))


    model = 'cnnlstm'
    n_views = 3 # no. of camera views

    input_resolution = 256

    scale = 0.5 # image pre-processing

    lr = 1e-4 # learning rate


    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)