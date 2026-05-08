import json
import torch
from util import nethook
from util.utility import ensure_file_directory
def load_data(cfg):
    if cfg.data.endswith(".json"):
        data_file=cfg.data_dir+"/"+cfg.data
    else:
        data_file = cfg.data_dir+"/"+ cfg.data+".json"
    with open(data_file, "r") as f:
        data = json.load(f)
    data=data[:cfg.num_edits]
    return data
def load_model(model,cfg):
    weights_dir = cfg.cache_dir + "/saved_weights"
    weights_file = weights_dir + "/{}/{}-{}-{}.pt".format(cfg.algs.name, cfg.data, cfg.load_name,
                                                          cfg.llms.name.replace("/", "-"))
    device=torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    weights=torch.load(weights_file)
    with torch.no_grad():
        for key, value in weights.items():
            weight=nethook.get_parameter(model,key)
            weight[...]=value.to(device)
    return model
def save_model(model,cfg):
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        ).cpu()
        for layer in cfg.llms.layers
    }
    weights_dir = cfg.cache_dir + "/saved_weights"
    weights_file = weights_dir + "/{}/{}-{}-{}.pt".format(cfg.algs.name, cfg.data, cfg.save_name,
                                                          cfg.llms.name.replace("/", "-"))
    ensure_file_directory(weights_file)
    torch.save(weights, weights_file)                         
