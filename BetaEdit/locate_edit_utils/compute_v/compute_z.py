from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from locate_edit_utils.repr_tools  import *
from util import nethook
from omegaconf import DictConfig
def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    cfg: DictConfig,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    lm_w, ln_f = (
        nethook.get_module(model, f"{cfg.llms.lm_head_module}").weight.T,
        nethook.get_module(model, cfg.llms.ln_f_module),
    )
    lm_b = nethook.get_parameter(model, f"{cfg.llms.lm_head_module}.bias")
    if lm_b is None:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)
    target_ids = tok(request["target_new"], return_tensors="pt").to(device)[
        "input_ids"
    ][0]
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts
    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(device)
    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, cfg.llms.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]
    loss_layer = max(cfg.llms.v_loss_layer, layer)
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=device)
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=device)
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None
    out_tuple=None
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        nonlocal out_tuple
        out_tuple=type(cur_out)==tuple
        if not out_tuple:                                          
            cur_out = (cur_out,)                                              
        if cur_layer == cfg.llms.layer_module_tmp.format(layer):
            if target_init is None:
                print("Recording initial value of v*")
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs)!=len(cur_out[0]):                            
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta
        if not out_tuple:
            cur_out = cur_out[0]
        return cur_out
    opt = torch.optim.Adam([delta], lr=cfg.llms.v_lr)
    nethook.set_requires_grad(False, model)
    for it in range(cfg.llms.v_num_grad_steps):
        opt.zero_grad()
        with nethook.TraceDict(
            module=model,
            layers=[
                cfg.llms.layer_module_tmp.format(loss_layer),
                cfg.llms.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits                       
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()
        if not out_tuple:
            output=tr[cfg.llms.layer_module_tmp.format(loss_layer)].output
        else:
            output=tr[cfg.llms.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1]!=rewriting_targets.shape[1]:
            output=torch.transpose(output, 0, 1)
        full_repr = output[:len(rewriting_prompts)]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = cfg.llms.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = cfg.llms.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break
        if it == cfg.llms.v_num_grad_steps - 1:
            break
        loss.backward()
        opt.step()
        max_norm = cfg.llms.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )
    return target
def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")
    return l_input.detach(), l_output.detach()
def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """
    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")
    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )
    return ret
