import os
import sys
import torch
import random
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from evals.evaluation import eval_algo
from datetime import datetime
from load import load_model,load_data,save_model
from util.utility import ensure_file_directory
import numpy as np
import time
import importlib
def apply_method_to_model(method):
    module_name = f"algs.{method}"
    function_name = f"apply_{method}_to_model"
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func
    except ImportError:
        raise ImportError(f"Module {module_name} not found.")
    except AttributeError:
        raise AttributeError(f"Function {function_name} not found in {module_name}.")
MODEL_PATH = {}
def set_random_seed(seed=42):
    torch.manual_seed(seed)                
    torch.cuda.manual_seed_all(seed)                
    torch.backends.cudnn.benchmark = False                                            
    torch.backends.cudnn.deterministic = True                              
    np.random.seed(seed)           
    random.seed(seed)               
    os.environ['PYTHONHASHSEED'] = str(seed)                
import hydra
from omegaconf import DictConfig, OmegaConf
def print_dict(dict):
    for key, value in dict.items():
        print(key, value)
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_random_seed(cfg.seed)
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    print("Start Loading model")
    model_name=cfg.llms.name
    model_name_or_path=MODEL_PATH.get(model_name,model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=cfg.model_dtype).to(device)
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    print("Loading model successfully")
    tok.pad_token = tok.eos_token
    apply_algo = apply_method_to_model(cfg.algs.name)
    data=load_data(cfg)
    if cfg.test_only:
        from evals.lweval import predicts
        pre_results_file = cfg.results_dir + "/{}/{}/{}".format(cfg.data, cfg.llms.name.replace("/", "-"), cfg.save_name)
        ensure_file_directory(pre_results_file)
        if not os.path.exists(pre_results_file) and cfg.pre_eval:
            start_time = time.time()
            print("Start Evaluating the Original Model")
            pre_metrics=eval_algo(cfg, model, tok, data)               
            end_time = time.time()
            hours = np.round((end_time - start_time) / 3600, 3)
            with open(pre_results_file, "w", encoding="utf-8") as f:
                f.write("\n\nEvaluation Took {} Hours".format(hours))
                f.write("\n\n")
                f.write("The Evaluation Results before Editing:")
                f.write("\n\n")
                json.dump(pre_metrics, f, ensure_ascii=False, indent=2)
                f.write("\n\n")
            print("End Evaluating the Original Model")
            if cfg.lw_eval:
                file="lw_eval/"+cfg.llms.name.replace("/", "-")+"/"+cfg.data+"/pred_lw_eval.npy"
                ensure_file_directory(file)
                np.save(file,np.array(predicts))
                predicts.clear()
        edited_model=load_model(model,cfg)
        print("Start Evaluating the Edited Model")
        post_metrics = eval_algo(cfg, edited_model, tok, data)
        post_results_file = cfg.results_dir + "/{}/{}/{}-{}".format(cfg.data, cfg.llms.name.replace("/","-"),cfg.algs.name, cfg.save_name)
        ensure_file_directory(post_results_file)
        with open(post_results_file, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg) + "\n\n")               
            f.write("The Evaluation Results after Editing:")
            f.write("\n\n")
            json.dump(post_metrics, f, ensure_ascii=False, indent=2)
            f.write("\n\n")
        print("End Evaluating the Edited Model")
        if cfg.lw_eval:
            file = "lw_eval/"+cfg.llms.name.replace("/", "-") + "/" + cfg.data+"/"+cfg.algs.name + "/pred_lw_eval.npy"
            ensure_file_directory(file)
            np.save(file, np.array(predicts))
    else:
        pre_metrics={}
        if cfg.debug_mode and cfg.num_edits<=500:
            pre_metrics = eval_algo(cfg, model, tok, data)                  
        edited_model = apply_algo(model,tok,data,cfg)
        if cfg.debug_mode:
            post_metrics = eval_algo(cfg, edited_model, tok, data)
            print("The Evaluation Results before Editing:")
            print_dict(pre_metrics)
            print("\n\n")
            print("The Evaluation Results after Editing:")
            print_dict(post_metrics)
        else:
            save_model(edited_model,cfg)
if __name__ == "__main__":
    main()
