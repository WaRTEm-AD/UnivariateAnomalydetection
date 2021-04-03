import yaml
import os

curr_folder = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_folder)
out_file = os.path.join(curr_folder, "default_params.yaml")

def dump_params(params, param_dump_path):
    with open(param_dump_path, 'w+') as fout:
        yaml.dump(params, fout, default_flow_style=False)

default_params = {
'resume_check': True, # Whether to check if a model is partially trained and resume. If false, it will start fresh regardless.(True)
'validation_split': 0.8, # the train - validation split. (Labels are not used. The method is still unsupervised.)
'input_data_folder': os.path.join(root_dir, 'data/'), # input data folder.
'dataset': 'input_data', # Name of input file. Dummy file name.
'rep_length':10, # We are working with a fixed representation length now.(10-for point,40=for sequence)
'num_inits': 1, # Number of models to learn. the model is trained these many times (each with a different weight initialization). To show robustness, you can use more inits and average.
'loss_path': os.path.join(root_dir, 'wartem/losses/'), # path to folder for loss values.
'model_path': os.path.join(root_dir, 'wartem/models/'), # path to folder which contains trained models.
'params_dump_path': os.path.join(root_dir, 'wartem/params_dump/'), # path to folder which contains the record of parameters used to train models.
'epochs':2000, # Maximum number of epochs to train for.(2000)
'verbosity': 0, # Verbosity for the training procedure. 0 suppresses the information about epochs, etc.
'learning_rate': 0.001, # learning rate for the optimizer.
'n_channels': 1, # Number of channels in the input. Since our input is univariate, this is 1. 
'num_warps': 50, # number of warpings created in a series = r, r is randomly sampled between 0 and 50% of series length, for each series.(50)
'batch_size': 32, # batch size for optimization. 
'shuffle': True, # whether to shuffle the IDs of datapoints each time. If true, then in each epoch, data is provided in a different order.(True)
'warping': 'copy', # warping to use : one of [copy, interpolation, copy_interpolation]
'point':True, # give whether point anomaly or sequence anomaly
'extra_verbose': 1,
'early_stopping': {
    'monitor': 'val_loss',
    'mode': 'min',
    'verbose': 1,
    'patience': 50,
    'min_delta': 0.01,
}, # input for early stopping callback. checkout early stopping in python.

'model_checkpoint': {
    'monitor': 'val_loss',
    'mode': 'min',
    'save_best_only': True,
}, # parameters for model checkpoint (checkpoints models while training so that you can resume from where you left off)
}

dump_params(default_params, out_file)
