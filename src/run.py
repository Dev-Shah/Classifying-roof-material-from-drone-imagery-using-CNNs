import sys
import json
import os
from training import *
from testing import *
from torch.cuda import is_available

dir_ = 'data'

type_ = sys.argv[1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if type_ == 'basic':
    with open('basic.json', 'r') as f:
        parameters = json.load(f)
    model_name = parameters['model_name']
    optimizer_name = parameters['optimizer_name']
    num_epochs = parameters['num_epochs']
    learning_rate = parameters['learning_rate']
    weight_decay = parameters['weight_decay']
    step_size = parameters['step_size']
    gamma = parameters['gamma']




    model = basic_training(dir_,model_name, optimizer_name, num_epochs, learning_rate, weight_decay, step_size, gamma, device)

    results = get_results(dir_, model, device, type_, weighted_average = False)

else:
    with open('snapshot.json', 'r') as f:
        parameters = json.load(f)
    model_name = parameters['model_name']
    optimizer_name = parameters['optimizer_name']
    num_epochs = parameters['num_epochs']
    num_cycles = parameters['num_cycles']
    lr_max = parameters['lr_max']
    weighted_average = parameters['weighted_average']

    models = snapshot_ensembling_training(dir_,model_name, optimizer_name, num_epochs, num_cycles, lr_max, device, weighted_average = weighted_average)

    get_results(dir_, models, device, type_, weighted_average)
