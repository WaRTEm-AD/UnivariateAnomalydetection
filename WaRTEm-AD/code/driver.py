#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from keras.models import load_model

import numpy as np
import utils
import layers

def get_parameters():
    curr_folder = os.path.dirname(os.path.abspath(__file__))

    # Load parameters from yaml file.
    default_params = utils.load_params(os.path.join(curr_folder, 'default_params.yaml'))
    override_params = utils.load_params(os.path.join(curr_folder, 'override_params.yaml'))
    
    # Parse commandline arguments.
    commandline_args = utils.parse_commandline_args()
    
    # Resolve params.
    params = utils.resolve_params(default_params, override_params, commandline_args)
    utils.choosy_print("Parameters resolved...", params)
    
    return params

def train_model(params):
    
    #%%
    # Set up data. 
    utils.choosy_print("Setting up data...", params)
    num_samples, series_length, df = utils.setup_data_files(params)
    # Add to params.
    params["num_samples"] = num_samples
    params["series_length"] = series_length
    
    utils.choosy_print("Data set up finished.", params)
    
    #%%
    # Validation split.
    ind = int(np.ceil(params["num_samples"] * params['validation_split']))
    indices=range(params['num_samples'])
    random.shuffle([indices])
    partition = {"train":list(indices[::ind]), "validation":list(indices[ind::])}
    
    params["partition"] = partition
    
    # Initialize architecture.
    network = utils.initialize_architecture(params)
    params["output_index"] = network.get_output_index()

    # Train required number of models.
    
    loss_dir = os.path.join(params['loss_path'], params["dataset"], params["warping"])
    model_dir = os.path.join(params['model_path'], params["dataset"], params["warping"])
    params_dump_dir = os.path.join(params['params_dump_path'], params["dataset"], params["warping"])
    
    # Set these directories up.
    utils.establish(loss_dir)
    utils.establish(model_dir)
    utils.establish(params_dump_dir)
    
    print("Dataset: ", params["dataset"])
    # Train required number of models.
    for i in range(params['num_inits']):
        print("Initialization: ", str(i), " | warping: ", params["warping"])

        # Set up paths.
        loss_path = os.path.join(loss_dir, str(i) + '.csv')
        model_path = os.path.join(model_dir, str(i) + '.h5')
        params_dump_path = os.path.join(params_dump_dir, str(i) + '.yaml')
        
        
        
        if not os.path.exists(loss_path):
            # Not previously completed.
            if os.path.exists(model_path) and params["resume_check"]:
                # has begun training. Not completed.
                print("Resuming training...")
                model = load_model(model_path, custom_objects={"custom_loss": utils.custom_loss,
			 	"MaxPoolingWithArgmax1D": layers.MaxPoolingWithArgmax1D,
				"MaxUnpooling1D": layers.MaxUnpooling1D})
            else:
                # Model has not begun training yet.
                print("Beginning training...")
    
                # Create autoencoder architecture.
                model = network.create_new_architecture(seed=i)
                model.save(model_path)
    
            # Data generators.
            train_generator = utils.select_generator('train', df, params, seed=i)
            validation_generator = utils.select_generator('validation', df, params, seed=i)
            
            # setup callbacks
            es = utils.setup_early_stopping(params)
            mc = utils.setup_model_checkpoint(model_path, params)
    
            # Train model.
    
            hist = model.fit_generator(
                generator=train_generator,
                validation_data=validation_generator,
                use_multiprocessing=False,
                callbacks=[es, mc],
                epochs=params['epochs'],
                verbose=params['verbosity'],
                )
            '''
            hist = model.fit_generator(
                generator=train_generator,
                validation_data=validation_generator,
                use_multiprocessing=False,
                epochs=params['epochs'],
                verbose=params['verbosity'],
                )
            ''' 
            print("Training done.")
            utils.choosy_print("Model saved.", params)
            
            utils.choosy_print("Storing loss values...", params)
            with open(loss_path, 'w+') as fout:
                for t, v in zip(hist.history['loss'], hist.history['val_loss']):
                    fout.write(str(t) + ',' + str(v) + '\n')
            utils.choosy_print("Losses stored.", params)
    
            # Delete model since this is not the best model. best model has automatically been written to file.
            del model
    
            # Dump the parameters to yaml file for future reference.
            utils.dump_params(params, params_dump_path)
            utils.choosy_print("Parameters stored.", params)
        else:
            # Training already done.
            utils.choosy_print("Initialization " + str(i) + " already completed. Moving on.", params)


if __name__ == "__main__":
    train_model(get_parameters())
