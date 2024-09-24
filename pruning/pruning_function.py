import torch 
import torch.nn as nn
from nni.compression.pruning import L1NormPruner
from nni.compression.speedup import ModelSpeedup
import copy



def prune_model(model, 
                sparse_ratio, 
                input_shape, 
                pruned_layer_types=['Linear','Conv2d','Conv3d','BatchNorm2d'], 
                exclude_layer_names=None,
                prunner_choice =  None):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    def find_last_layer_name_and_module(model):
        last_name, last_module = None, None
        for name, module in model.named_modules():
            if not list(module.children()):  # If the current module has no children
                last_name, last_module = name, module
        return last_name, last_module
    
    config_list = [{}]
    config_list [0]['sparse_ratio'] = sparse_ratio


    config_list[0]['op_types'] =pruned_layer_types
    last_layer_name, _ = find_last_layer_name_and_module(model)
    if not exclude_layer_names:
     config_list[0]['exclude_op_names'] = [last_layer_name]
     print('NOTE: the last layer of the model was not provided and was automatically detected. If any error occur it')
    else:
        config_list[0]['exclude_op_names'] = exclude_layer_names

    if not prunner_choice:
        prunner_choice = L1NormPruner

    pruner = prunner_choice(model, config_list)
    _, masks = pruner.compress()
    pruner.unwrap_model()
    ModelSpeedup(model, torch.rand(input_shape).to(device), masks).speedup_model()
    print(model)
    return copy.deepcopy(model)
