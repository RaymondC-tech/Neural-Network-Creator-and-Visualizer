import torch.nn as nn
import json
from pathlib import Path

_ACTIVATIONS = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        None: nn.Identity,    # maps JSON null -> no activation
    }


def build_model(input_dim: int, spec_path: Path | str) -> nn.Module:
    """
    Read the Json, return a nn.Sequential.
    imput_dim: number of numeric features coming from the Dataset
    spec_path: path ot the JSON file
    """
    #read and parse the json folder
    spec = json.loads(Path(spec_path).read_text())

    #pulling fields out of the spec
    layers_cfg = spec["layers"] 
    out_units = spec.get("output_units", 1)  # numbner of neurons on the final layer
    out_act_name = spec.get("output_activation", None)  #final layer activatoin function

    modules = []
    prev_dim = input_dim

    #2 hidden layer loop
    for cfg in layers_cfg:
        units = cfg["units"]  # number of neurons in that unit
        act = cfg["activation"].lower()  #activtion function for that layer
        modules.append(nn.Linear(prev_dim, units))  #incoming matrix and neurons u want in this layer / shape coming out
        modules.append(_ACTIVATIONS[act]())
        prev_dim = units

    #3. ouput layers
    modules.append(nn.Linear(prev_dim, out_units))
    modules.append(_ACTIVATIONS[out_act_name]())

    return nn.Sequential(*modules) # the asterik tells Python to take each element of the list and feed it to the function one-by-one
    