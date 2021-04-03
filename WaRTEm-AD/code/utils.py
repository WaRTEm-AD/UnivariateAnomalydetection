import yaml
import sys
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

import layers
from keras.layers import Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model, Model
from TwinArchitecture import Architecture
from data_generator import InterpolationGenerator, CopyGenerator, CopyInterpolationGenerator

def custom_loss(y_true, y_pred):
    return y_pred

def choosy_print(st, params):
    if params["extra_verbose"] == 1:
        print(st)

def parse_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--dataset", help="filename (without extension)", type=str)
    parser.add_argument("-n", "--num_inits", help="number of initializations to train with", type=int)
    parser.add_argument("-w", "--warping", help="warping to use", type=str)
    args = parser.parse_args()

    return vars(args)

def load_params(file_path):
    with open(file_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
    return params


def resolve_params(default_params, override_params, commandline_args):
    params = default_params
    
    if override_params is not None:
        for key, value in override_params.items():
            params[key] = value

    if commandline_args is not None:
        for key, value in commandline_args.items():
            if value is not None:
                params[key] = value

    # By default, work with a particular filename. in case a file name is input through the commandline or override, use that file instead.
    params["input_data_file"] = os.path.join(params["input_data_folder"], params["dataset"] + '.csv')
    return params

def establish(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_data_files(params):
    "Reads data file. returns data properties and data."

    df = pd.read_csv(params["input_data_file"], header=None)
    num_samples = df.shape[0]
    series_length = df.shape[1]
    
    return num_samples, series_length, df

def initialize_architecture(params):
    return Architecture(
        series_length=params["series_length"],
        learning_rate=params["learning_rate"],
        rep_length=params["rep_length"],
        n_channels=params["n_channels"],
        )

def select_generator(index, df, params, seed=1234):
    generators = {
            'copy': CopyGenerator,
            'interpolation': InterpolationGenerator,
            'copy_interpolation': CopyInterpolationGenerator,
            }

    # select generator. 
    generator = generators[params["warping"]]

    if index == "train":
        return generator(
                params["partition"]["train"],
                params["series_length"],
                df=df,
                num_warps=params["num_warps"],
                batch_size=params["batch_size"], 
                n_channels=params["n_channels"],
                shuffle=params["shuffle"],
                seed=seed,
                params["point"],
                )
    else:
        return generator(
                params["partition"]["validation"],
                params["series_length"],
                df=df,
                num_warps=params["num_warps"],
                batch_size=params["batch_size"], 
                n_channels=params["n_channels"],
                shuffle=False,
                seed=seed,
                params["point"],
                )

def setup_early_stopping(params):
    return EarlyStopping(**params["early_stopping"])

def setup_model_checkpoint(model_path, params):
    return ModelCheckpoint(model_path, **params["model_checkpoint"])

def dump_params(params, param_dump_path):
    with open(param_dump_path, 'w+') as fout:
        yaml.dump(params, fout, default_flow_style=False)

def get_representations(model_path, datafile, output_index=[10,11]):
    "wrapper for get_reps function to handle the data processing."
    df = pd.read_csv(datafile,header=None)
    X = df.values
    X = np.expand_dims(X, axis=-1)

    return get_reps(model_path, X, output_index)

def get_reps(model_path, X, output_index):
    "use saved model to generate representations."
    model = load_model(model_path, custom_objects={"custom_loss": custom_loss,
			 	"MaxPoolingWithArgmax1D": layers.MaxPoolingWithArgmax1D,
				"MaxUnpooling1D": layers.MaxUnpooling1D})
    
     # Intermediate model to extract representation
    inputs_1 = model.get_layer(index=0).input
    inputs_2 = model.get_layer(index=1).input
    inputs = [inputs_1, inputs_2]
    
    outputs_1 = model.get_layer(index=output_index[0]).output
    outputs_2 = model.get_layer(index=output_index[1]).output
    
    def mean_layer(tensors):
        return (tensors[0] + tensors[1])/2
    
    outputs = Lambda(mean_layer)([outputs_1, outputs_2])
    
    intermediate_model = Model(inputs=inputs, outputs=outputs)
    intermediate_model.summary()
    
    
    # Extract representations
    representations = intermediate_model.predict([X, X])
    print("representations",np.shape(representations))
    return representations


def get_reconstructions(model_path, datafile):
    "wrapper for get_reps function to handle the data processing."
    df = pd.read_csv(datafile,header=None)
    print(len(df))
    X = df.values
    print("X shape initial",np.shape(X))
    X = np.expand_dims(X, axis=-1)
    print("X shape",np.shape(X))
    return get_recons(model_path, X)

def get_recons(model_path, X):
    "use saved model to generate representations."
    model = load_model(model_path, custom_objects={"custom_loss": custom_loss,
			 	"MaxPoolingWithArgmax1D": layers.MaxPoolingWithArgmax1D,
				"MaxUnpooling1D": layers.MaxUnpooling1D})
    
     # Intermediate model to extract representation
    inputs_1 = model.get_layer(index=0).input
    inputs_2 = model.get_layer(index=1).input
    inputs = [inputs_1, inputs_2]
    
    outputs_1 = model.get_layer(index=50).output
    outputs_2 = model.get_layer(index=51).output
    
    def mean_layer(tensors):
      for i in range(len(X)):
        if i%2==0:
          outputs=tensors[1]
        
        else:
          outputs=tensors[0]
      return outputs
    #return (tensors[0] + tensors[1])/2
    
    outputs = Lambda(mean_layer)([outputs_1, outputs_2])
    
    intermediate_model = Model(inputs=inputs, outputs=outputs)
    
    # Extract representations
    reconstructions = intermediate_model.predict([X, X])
    print("reconstructions",np.shape(reconstructions))
    return reconstructions
