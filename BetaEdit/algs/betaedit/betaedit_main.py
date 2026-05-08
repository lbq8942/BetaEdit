import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from torch import device
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.layer_stats import get_cov
from util import nethook
from util.generate import generate_fast
from omegaconf import DictConfig
from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from algs.memitf.fast import load_prior
from evals.evaluation import eval_every
import math
# Cache variable(s)
Ps=[]
covs=[]
Us=[]
CONTEXT_TEMPLATES_CACHE = None
from util.utility import ensure_file_directory

def load_cov(cfg,model,tok):
    layers=cfg.llms.layers
    for i, layer in enumerate(layers):
        cov=get_cov(
            cfg,
            model,
            tok,
            layer,
            cfg.llms.mom2_dataset,
            cfg.llms.mom2_n_samples,
            cfg.llms.mom2_dtype,
            force_recompute=False,
        )
        covs.append(cov)

def load_project(cfg):
    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n\nLAYER {layer}\n")
        Ppathi = cfg.cache_dir + "/null_space_project/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) + ".pt"
        Pi = torch.load(Ppathi,map_location="cpu")
        Ps.append(Pi)


import torch


class ProjectionUpdater:
    def __init__(self, current_P,update_interval=1000):
        """
        Initialize the projection updater.

        Args:
            update_interval (int): Number of steps after which to update P
        """
        self.update_interval = update_interval
        self.last_update_step = 0
        self.current_P = current_P.cpu()

    def get_project(self, cfg, cache, current_step,cov):
        """
        Get projection matrix P, updating it if necessary.

        Args:
            cfg: Configuration object with algs.nullspace_threshold
            cache: Current cache tensor/matrix
            current_step: Current step counter

        Returns:
            P: Projection matrix
        """
        # Check if we need to update
        device=cache.device
        if (current_step - self.last_update_step) >= self.update_interval:
            Proteced_Set=(cache/current_step+cfg.algs.lambda1*cov.to(device))/(cfg.algs.lambda1+1)
            self._update_P(cfg, Proteced_Set)
            self.last_update_step = current_step#

        return self.current_P.to(device)

    def _update_P(self, cfg, cache):
        """
        Compute the projection matrix P from cache.

        Args:
            cfg: Configuration object
            cache: Cache tensor/matrix
        """
        U, S, _ = torch.linalg.svd(cache, full_matrices=False)
        threshold = cfg.algs.nullspace_threshold
        small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]

        # Print the number of small singular values (optional)
        # print("the number of small singular values", len(small_singular_indices))

        # Compute P
        self.current_P = U[:, small_singular_indices] @ U[:, small_singular_indices].T
        self.current_P = self.current_P.cpu()

    def force_update(self, cfg, cache):
        """
        Force an immediate update of P regardless of step counter.

        Args:
            cfg: Configuration object
            cache: Current cache tensor/matrix
        """
        self._update_P(cfg, cache)
        self.last_update_step = 0  # Reset counter

    def get_last_update_step(self):
        """Return the step number of the last update."""
        return self.last_update_step

    def get_update_interval(self):
        """Return the current update interval."""
        return self.update_interval

    def set_update_interval(self, new_interval):
        """
        Set a new update interval.

        Args:
            new_interval (int): New update interval in steps
        """
        self.update_interval = new_interval


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def get_fc_dim(model,cfg):
    W_out = nethook.get_parameter(model, f"{cfg.llms.rewrite_module_tmp.format(1)}.weight")
    fc_dim=W_out.shape[0] if W_out.shape[0]>W_out.shape[1] else W_out.shape[1]
    return fc_dim
# norms=[]
def apply_betaedit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig
):
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    requests_copy=deepcopy(requests)
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        requests[i]["target_new"] = " " + request["target_new"]
    layers=cfg.llms.layers
    # compute the null space project P.
    for i, layer in enumerate(layers):
        Ppathi = cfg.cache_dir + "/null_space_project/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) + ".pt"
        ensure_file_directory(Ppathi)
        if not os.path.exists(Ppathi):#then compute
            print("The null-space projection matrix P for model {} layer {} "
                  "does not exist and now calculate.".format(cfg.llms.name.replace("/","-"), layer))
            Pi = get_project(model, tok, layer, cfg)
            torch.save(Pi.cpu(), Ppathi)
    load_project(cfg)
    PUs=[ProjectionUpdater(P,cfg.algs.P_freq) for P in Ps]
    load_cov(cfg,model,tok)

    fc_dim=get_fc_dim(model,cfg)
    cache_c=[torch.zeros(fc_dim,fc_dim,device=device) for i in range(len(layers))]#默认就在cpu上面。


    import time
    start=time.time()
    count=0
    metrics_all=[]
    for requests_chunks in chunks(requests, cfg.bs):
        corrupt=batch_edit(cfg,model,tok,requests_chunks,device,cache_c,PUs,count)
        count+=cfg.bs
        if corrupt:
            break
        if count%cfg.eval_every==0:
            print()
            print("##########Evaluating#########")
            metrics=eval_every(cfg,model,tok,requests_copy[:count],count)
            metrics_all.append(metrics)
            threshold=0.01
            break_count = sum(1 for v in metrics.values() if isinstance(v, (int, float)) and v < threshold)
            if break_count>=3:
                break



    # total_time=time.time()-start
    # zzz_file=cfg.cache_dir+"/deltau/{}-{}-{}.tensor".format(cfg.algs.name,cfg.llms.name.replace("/","-"),cfg.data)
    # ensure_file_directory(zzz_file)
    # # torch.save(norms,zzz_file)
    # # 写入一行，带换行符.
    # print("{}/{}/{}/num_edits:{}/bs:{}/time:{}\n".format(cfg.data,cfg.algs.name,cfg.llms.name,
    #                                                         cfg.num_edits,cfg.bs,total_time))
    # with open('times.txt', 'a', encoding='utf-8') as f:
    #     f.write("{}/{}/{}/num_edits:{}/bs:{}/time:{}\n".format(cfg.data,cfg.algs.name,cfg.llms.name,
    #                                                         cfg.num_edits,cfg.bs,total_time))
    return model


def batch_edit(cfg,model,tok,requests,device,cache_c,PUs,count):
    # deltas = {}
    # Retrieve weights that user desires to change
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in cfg.llms.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = cfg.llms.layers[-1]
    z_list = []
    for request in requests:
        cur_z = compute_z(
            model,
            tok,
            request,
            cfg,
            z_layer,
            context_templates,
        )
        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=1)
    corrupt=False
    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n\nLAYER {layer}\n")
        Pi = PUs[i].get_project(cfg,cache_c[i],count,covs[i])
        # print(layer_ks.sum(),layer_ks.mean(),layer_ks.std())
        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=cfg.llms.layer_module_tmp,
            fact_token_strategy=cfg.llms.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(cfg.llms.layers) - i)  # Distribute residual across layers
        upd_type=torch.float
        # import time
        # start=time.time()
        upd_matrix = torch.linalg.solve(
                Pi @ (layer_ks @ layer_ks.T + cache_c[i]+cfg.algs.lambda1*covs[i].to(device)) +
                cfg.algs.lambda2*torch.eye(layer_ks.shape[0], dtype=upd_type,device=device),
            Pi @ layer_ks.to(upd_type) @ resid.T
        )
        cache_c[i] = cache_c[i] + layer_ks @ layer_ks.T
        # Adjust update matrix shape
        weight_name = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            if torch.linalg.norm(upd_matrix)/math.sqrt(layer_ks.shape[1]) > 20:
                corrupt=True
                break
            else:
                weights[weight_name][...] = weights[weight_name] + upd_matrix
            # deltas[weight_name] = upd_matrix
            # if len(upds)<=i:
            #     upds.append(upd_matrix)
            # else:
            #     upds[i]+=upd_matrix

            # if upds[i].shape[1] == Us[i][0].shape[0]:
            #     norms.append(torch.norm(upds[i] @ Us[i][0].to(device)).item())#主方向损失。
            #     norms.append(torch.norm(upds[i] @ Us[i][1].to(device)).item())#次要方向损失。
            #     norms.append(torch.trace(upds[i] @ covs[i].to(device)@upds[i].T).item())#完整损失。
            # else:
            #     norms.append(torch.norm(upds[i].T @ Us[i][0].to(device)).item())
            #     norms.append(torch.norm(upds[i].T @ Us[i][1].to(device)).item())
            #     norms.append(torch.trace(upds[i].T @ covs[i].to(device)@upds[i]).item())
        # Pi.cpu()
        # for x in [layer_ks, cur_zs, targets]:
        #     x.cpu()
        #     del x
        # torch.cuda.empty_cache()
    return corrupt
    # if cfg.algs.add_old_keys:
    #     for i, layer in enumerate(cfg.llms.layers):
    #         layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
    #         cache_c[i, :, :] += (layer_ks @ layer_ks.T).cpu()


def get_project(model, tok, layer, cfg):
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    force_recompute = False
    cov = get_cov(
        cfg,
        model,
        tok,
        layer,
        cfg.llms.mom2_dataset,
        cfg.llms.mom2_n_samples,
        cfg.llms.mom2_dtype,
        force_recompute=force_recompute,
    )
    U, S, _ = torch.linalg.svd(cov.to(device), full_matrices=False)
    threshold = cfg.algs.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    # print("the number of small singular values",len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


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
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE