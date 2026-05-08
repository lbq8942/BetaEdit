"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
import typing
from itertools import chain
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from evals.lweval import lw_eval
from evals.lbqeval import lbq_eval
def get_prompt_target_pairs(tok,model,prompt,target):          
    prompts, targets=[],[]
    target_tok = tok(" " + target,add_special_tokens=False)["input_ids"]                       
    for i in range(len(target_tok)):
        prompts.append(prompt + tok.decode(target_tok[:i]))
        targets.append(tok.decode(target_tok[i]))
    return prompts,targets
def eval_zsre(
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
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    subject, target_new, neighborhood_answer = (
        record[x] for x in ["subject", "target_new","neighborhood_prompts_answers"]
    )
    rewrite_prompts = [record["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    rewrite_prompts,rewrite_targets=get_prompt_target_pairs(tok,model,rewrite_prompts[0],target_new)
    paraphrase_prompts,paraphrase_targets=get_prompt_target_pairs(tok,model,paraphrase_prompts[0],target_new)
    neighborhood_prompts,neighborhood_targets=get_prompt_target_pairs(tok,model,neighborhood_prompts[0],neighborhood_answer[0])
    prompts=rewrite_prompts+paraphrase_prompts+neighborhood_prompts
    targets=rewrite_targets+paraphrase_targets+neighborhood_targets
    metrics_detail = test_batch_prediction_acc(model, tok, prompts, targets)                                          
    cut_offs=np.cumsum([0,len(rewrite_targets),len(paraphrase_targets),len(neighborhood_targets)])
    keys=["rewrite_prompts_correct","paraphrase_prompts_correct","neighborhood_prompts_correct"]
    strict_keys=["rewrite_strict_correct","paraphrase_strict_correct","neighborhood_strict_correct"]
    metrics={}
    for i in range(len(keys)):
        start=cut_offs[i]
        end=cut_offs[i+1]
        metrics[keys[i]]=np.mean(metrics_detail[start:end]).item()
        metrics[strict_keys[i]]= np.min(metrics_detail[start:end]).astype(np.int32).item()
    if cfg.lbq_eval:
        metrics_lbq = lbq_eval(record, cfg, model, tok,q_test=False)
        metrics = metrics | metrics_lbq
    if cfg.lw_eval:
        metrics_lw=lw_eval(record,cfg,model,tok)
        metrics=metrics | metrics_lw
    return metrics
def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    device=next(model.parameters()).device
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**prompt_tok).logits                       
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1                    
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)                  
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)                            
        ans = torch.argmax(gathered, dim=1)                      
        correct_id = tok(target, padding=True, return_tensors="pt",
                         add_special_tokens=False).to(device)["input_ids"]                                  
        correct_id = correct_id[:, 0].squeeze()
        return (ans == correct_id).detach().cpu().numpy().tolist()
