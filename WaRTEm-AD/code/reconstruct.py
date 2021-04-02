#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
from utils import load_params, get_representations, establish, get_reconstructions

"""
This file can be used to extract representations from any saved model (that was created using the code in this folder).
Representations are saved to corresponding folder.
usage:
    python extract_representations.py <dataset> <warping> <model_number>
example:
    python extract_representations.py FordA interpolation 3
"""

def extract_reconstruct():
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(curr_folder)
    
    # Expects three arguments (program name plus 3 more): dataset, warping, model_number
    if len(sys.argv) != 4 :
        print("incorrect arguments! Aborting..")
        sys.exit()
    
    dataset = sys.argv[1]
    warping = sys.argv[2]
    model_num = sys.argv[3]
    
    model_path = os.path.join(root_dir, "wartem", "models", dataset, warping, str(model_num) + '.h5')
    params_path = os.path.join(root_dir, "wartem", "params_dump", dataset, warping, str(model_num) + '.yaml')
    
    if not os.path.exists(model_path) or not os.path.exists(params_path):
        print("Model/parameters have not been stored. Cannot extract reconstuctions.")
        sys.exit()    

    # Load saved params.
    params = load_params(params_path)
    
    # Data file that was used for training the model.
    input_data_file = params['input_data_file']
    
    reconstruct = get_reconstructions(model_path, input_data_file)
    
    reconstruct=np.reshape(reconstruct,[reconstruct.shape[0],reconstruct.shape[1]])
    print("shape to write",np.shape(reconstruct))
    # write representations to file.
    output_data_folder = os.path.join(root_dir, "wartem", "reconstruct", dataset, warping)
    establish(output_data_folder)
    output_data_file = os.path.join(output_data_folder, str(model_num) + '.csv')
    
    np.savetxt(output_data_file, reconstruct, delimiter=',',fmt='%10.5f')

if __name__ == "__main__":
    extract_reconstruct()
