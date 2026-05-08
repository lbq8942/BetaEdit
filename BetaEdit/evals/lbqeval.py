"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""
from util.generate import generate_fast
import typing
from itertools import chain
import nltk
import numpy as np
import scipy
import torch
def eval_counterfact(cfg,model,tok,prob_prompts,which_correct,
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
def build_prompt(question):
    template=('Q: Who is the current US president as of 2025? \nA: Joe Biden\n'
              'Q:{} \nA:')
    prompt=template.format(question)
    return prompt
def eval_zsre(model, tok, prompts: typing.List[str], target):
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
    metrics_detail=(ans == correct_id).detach().cpu().numpy().tolist()
    cut_offs = np.cumsum([0, len(target)])
    keys = ["rewrite_cloze_correct"]
    metrics = {}
    for i in range(len(keys)):
        start = cut_offs[i]
        end = cut_offs[i + 1]
        metrics[keys[i]+"_strict"] = np.min(metrics_detail[start:end]).astype(np.int32).item()
        metrics[keys[i]] = np.mean(metrics_detail[start:end]).item()
    return metrics
def get_prompt_target_pairs(tok,model,prompt,target):          
    prompts, targets=[],[]
    target_tok = tok(" " + target,add_special_tokens=False)["input_ids"]                       
    for i in range(len(target_tok)):
        prompts.append(prompt + tok.decode(target_tok[:i]))
        targets.append(tok.decode(target_tok[i]))
    return prompts,targets
def lbq_eval(record,cfg,model,tok,q_test=True):
    if not q_test:
        cloze=record["cloze"]
        target=record["target_new"]
        cloze_prompts,cloze_targets=get_prompt_target_pairs(tok, model, cloze, target)
        metrics=eval_zsre(model,tok,cloze_prompts,cloze_targets)
    else:
        question=record["question"]
        questions=[build_prompt(question),question]         
        prob_prompts=[[questions[0]],[questions[1]]]       
        which_correct=[[0],[0]]                     
        target_new=record["target_new"]
        target_true=record["target_true"]
        metrics=eval_counterfact(cfg,model,tok,prob_prompts,which_correct,target_new,target_true
                                   ,keys=["rewrite_q1","rewrite_q2"])
    return metrics
