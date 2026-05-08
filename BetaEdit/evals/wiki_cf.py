"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""
import typing
from itertools import chain
import nltk
import numpy as np
import scipy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util.generate import generate_fast
from util.perplexity import perplexity
from omegaconf import DictConfig
from evals.lweval import lw_eval
def compute_probs_correct(cfg,model,tok,prob_prompts,which_correct,
                          target_new,target_true,keys):
    probs, targets_correct = test_batch_prediction(
        cfg,
        model,
        tok,
        list(chain(*prob_prompts)),                               
        list(chain(*which_correct)),
        target_new,
        target_true,
    )
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    ret_probs=summarize(ret_probs, which_correct)
    ret = {
              f"{key}_probs": ret_probs[i]
              for i, key in enumerate(keys)
          } | {
              f"{key}_correct": ret_corrects[i]
              for i, key in enumerate(keys)
          }
    metrics=replace_tf_with_acc(ret)
    return metrics
def replace_tf_with_acc(metrics):
    for key in metrics:
        metrics[key]=np.mean(metrics[key]).item()
    return metrics
def summarize(kinds,labels):
    all_preds=[]
    for i in range(len(kinds)):
        kind=kinds[i]
        preds=[]
        for j in range(len(kind)):
            result=kind[j]
            label=labels[i][j]
            if label==0:
                eval_pred=result["target_new"]<result["target_true"]
            else:
                eval_pred=result["target_new"]>result["target_true"]
            preds.append(eval_pred)
        all_preds.append(preds)
    return all_preds
def eval_wiki_cf(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.
    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :return: Dictionary containing rewriting metrics
    """
    subject, target_new, target_true = (
        record[x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["prompt"].format(subject)]
    prob_prompts = [
        rewrite_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
    ]
    metrics=compute_probs_correct(cfg,model,tok,prob_prompts,which_correct,target_new,target_true
                                   ,keys=["rewrite_prompts"])
    if cfg.lw_eval:
        metrics_lw=lw_eval(record,cfg,model,tok)
        metrics=metrics | metrics_lw
    return metrics
def test_batch_prediction(
    cfg,
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """
    device=next(model.parameters()).device
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"                      
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)
    a_tok, b_tok = (tok(f" {n}",add_special_tokens=False)["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    with torch.no_grad():
        logits = model(**prompt_tok).logits
    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []
    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)
    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct
