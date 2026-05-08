import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from locate_edit_utils.layer_stats import get_cov
from util import nethook
from util.generate import generate_fast
from omegaconf import DictConfig
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
Ps=[]                                              
CONTEXT_TEMPLATES_CACHE = None
from util.utility import ensure_file_directory
def compute_v_star(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig
):
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        requests[i]["target_new"] = " " + request["target_new"]
    context_templates = get_context_templates(model, tok)
    z_layer = cfg.llms.layers[-1]
    for request in requests:
        cur_z = compute_z(
            model,
            tok,
            request,
            cfg,
            z_layer,
            context_templates,
        )
        torch.save()
def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]                                   
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")
    return CONTEXT_TEMPLATES_CACHE
import yaml
import argparse
import hydra
from main import MODEL_PATH
from load import load_data
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg):
    print("Start Loading model")
    model_name=cfg.llms.name
    model_name_or_path=MODEL_PATH.get(model_name,model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=cfg.model_dtype).to(device)
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    print("Loading model successfully")
    tok.pad_token = tok.eos_token
    data=load_data(cfg)
    compute_v_star(model,tok,data,cfg)
if __name__ == "__main__":
    main()
