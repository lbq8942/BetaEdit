import typing
import numpy as np
import scipy
import torch
from util.generate import generate_fast
predicts=[]
def check_next_word_binary(texts, target_words,class1="true",class2="false"):
    predictions=[]
    for i in range(len(texts)):
        text=texts[i]
        target_word=target_words[i]
        words = text.lower().split()           
        target_word=min(len(words)-1,target_word)        
        if class1 == words[target_word][:len(class1)]:
            predictions.append(0)
        elif class2 == words[target_word][:len(class2)]:
            predictions.append(1)
        else:
            predictions.append(-1)
    return predictions
def build_qwen_tf_prompt(prompt):
    qwen_prompt=(f"<|im_start|>system\nYou are a fact-checker. "
                 f"Answer only 'True' or 'False'.<|im_end|>\n"
    f"<|im_start|>user\n{prompt}<|im_end|>"
    f"\n<|im_start|>assistant\n")
    return qwen_prompt
def build_llama38b_tf_prompt(prompt):
    llama_prompt=(f"<|begin_of_sentence|>system\nYou are a fact-checker. "
                    f"Answer only 'True' or 'False'.<|end_of_sentence|>\n"
              f"<|begin_of_sentence|>user\n{prompt}<|end_of_sentence|>"
              f"\n<|begin_of_sentence|>assistant\n")
    return llama_prompt
def build_qwen_ab_prompt(question,target_new, target_true,statement=False):
    if statement:
        prompt = ("<|im_start|>system\nThere are two statements labeled A and B. "
                    "As of 2025-08-29, decide which one is more likely to be true based"
                    " on your general world knowledge. Reply with only A or B—no other"
                    " text, no punctuation, and no explanation.<|im_end|>\n"
                    "<|im_start|>user\nStatements:\n A: {}\nB: {}<|im_end|>"
                    "\n<|im_start|>assistant\n")
        first_prompt = prompt.format(target_new, target_true)
        second_prompt = prompt.format(target_true, target_new)
    else:
        prompt=("There is a question together with two possible answer candidates"
                   " marked with 'A' and 'B' respectively. Based on your knowledge,"
                   " please determine which candidate is the better answer to the "
                   "question. Reply with only A or B.\nQuestion: {}\nCandidate A: {}."
                   "\nCandidate B: {}.\nAnswer:")
        first_prompt=prompt.format(question,target_new, target_true)
        second_prompt=prompt.format(question,target_true, target_new)
    return first_prompt, second_prompt
def build_llama38b_ab_prompt(question,target_new, target_true,statement=False):
    if statement:
        prompt = ("<|begin_of_sentence|>system\nThere are two statements labeled A and B. "
                    "As of 2025-08-29, decide which one is more likely to be true based"
                    " on your general world knowledge. Reply with only A or B—no other"
                    " text, no punctuation, and no explanation.<|end_of_sentence|>\n"
                    "<|begin_of_sentence|>user\nStatements:\n A: {}\nB: {}<|end_of_sentence|>"
                    "\n<|begin_of_sentence|>assistant\n")
        first_prompt = prompt.format(target_new, target_true)
        second_prompt = prompt.format(target_true, target_new)
    else:
        prompt=("There is a question together with two possible answer candidates"
                   " marked with 'A' and 'B' respectively. Based on your knowledge,"
                   " please determine which candidate is the better answer to the "
                   "question. Reply with only A or B.\nQuestion: {}\nCandidate A: {}."
                   "\nCandidate B: {}.\nAnswer:")
        first_prompt=prompt.format(question,target_new, target_true)
        second_prompt=prompt.format(question,target_true, target_new)
    return first_prompt, second_prompt
def test_a_b(cfg,model,tok,ab_prompts,which_correct,key,question=None,statement=False):
    target_new,target_true=ab_prompts
    if "llama" in cfg.llms.name.replace("/","-").lower():
        prompts=build_llama38b_ab_prompt(question,target_new,target_true,statement=statement)
    elif "qwen" in cfg.llms.name.replace("/","-").lower():
        prompts=build_qwen_ab_prompt(question,target_new,target_true,statement=statement)
    suffix=["_first","_second","third","fourth"]
    keys=[key+suffix[i] for i in range(len(prompts))]
    assert len(prompts)==len(keys)
    n_gen_per_prompt=5                                                   
    lens=np.array([len(prompt.split()) for prompt in prompts])
    lens=np.repeat(lens.reshape(-1,1),n_gen_per_prompt,axis=1).reshape(-1)
    inp_tok = tok(prompts, padding=True, return_tensors="pt")
    max_out_len=inp_tok["input_ids"].shape[1]+5
    gen_texts=generate_fast(
            model,
            tok,
            prompts,
            n_gen_per_prompt=n_gen_per_prompt,
            max_out_len=max_out_len,
    )
    predictions=check_next_word_binary(gen_texts,lens,class1="a",class2="b")
    predictions=np.array(predictions).reshape(-1,n_gen_per_prompt)
    predicts.extend(list(predictions[:,0]))
    gts=np.repeat(np.array(which_correct).reshape(-1,1),n_gen_per_prompt,axis=1)
    acc=np.mean(predictions==gts,axis=1)
    metrics={}
    for i in range(len(keys)):
        metrics[keys[i]+"_next_acc"]=acc[i].item()
        metrics[keys[i]+"_next_nab_prob"]=np.mean(predictions[i]==-1).item()
    return metrics                      
def test_true_false(cfg,model,tok,prompts,which_correct,key):
    if "llama" in cfg.llms.name.replace("/","-").lower():
        prompts=[build_llama38b_tf_prompt(prefix) for prefix in prompts]
    elif "qwen" in cfg.llms.name.replace("/","-").lower():
        prompts=[build_qwen_tf_prompt(prefix) for prefix in prompts]
    suffix=["_first","_second","_third","_fourth"]
    keys=[key+suffix[i] for i in range(len(prompts))]
    assert len(prompts)==len(keys)
    n_gen_per_prompt=5                                                   
    lens=np.array([len(prompt.split()) for prompt in prompts])
    lens=np.repeat(lens.reshape(-1,1),n_gen_per_prompt,axis=1).reshape(-1)
    inp_tok = tok(prompts, padding=True, return_tensors="pt")
    max_out_len=inp_tok["input_ids"].shape[1]+5
    gen_texts=generate_fast(
            model,
            tok,
            prompts,
            n_gen_per_prompt=n_gen_per_prompt,
            max_out_len=max_out_len,
    )
    predictions=check_next_word_binary(gen_texts,lens)
    predictions=np.array(predictions).reshape(-1,n_gen_per_prompt)
    predicts.extend(list(predictions[:,0]))
    gts=np.repeat(np.array(which_correct).reshape(-1,1),n_gen_per_prompt,axis=1)
    acc=np.mean(predictions==gts,axis=1)
    metrics={}
    for i in range(len(keys)):
        metrics[keys[i]+"_next_acc"]=acc[i].item()
        metrics[keys[i]+"_next_ntf_prob"]=np.mean(predictions[i]==-1).item()
    return metrics                      
def replace_tf_with_detailed_acc(metrics):
    suffixes=["first","second","third","fourth"]
    new_metrics={}
    for key in metrics:
        if key!="rewrite_tf_probs" and key!="rewrite_tf_correct":
            continue
        for i in range(len(suffixes)):
            new_metrics[key+"_"+suffixes[i]]=int(metrics[key][i])
    return new_metrics
def test_true_false_prob(                                                
        cfg,
        model,
        tok,
        prefixes: typing.List[str],
        which_correct: list,
        target_new: str,
        target_true: str,
        key: str,
):
    if "llama" in cfg.llms.name.replace("/","-"):
        prefixes=[build_llama38b_tf_prompt(prefix) for prefix in prefixes]
    elif "qwen" in cfg.llms.name.replace("/","-"):
        prefixes=[build_qwen_tf_prompt(prefix) for prefix in prefixes]
    device = next(model.parameters()).device
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        prefixes,
        padding=True,
        return_tensors="pt",
    ).to(device)
    if cfg.llms.name=="meta-llama/Llama-3-8B-Instruct":                            
        a_tok, b_tok = (tok(f"{n}")["input_ids"] for n in [target_new, target_true])
    else:                 
        a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    if 'llama' in model.config._name_or_path.lower():
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [lengths - 1 for lengths in prefix_lens]
    with torch.no_grad():
        logits = model(**prompt_tok).logits
    if 'llama' in model.config._name_or_path.lower():
        logits = logits[:, 1:, :]
    a_probs=np.zeros((logits.size(0),), dtype=np.float32)
    b_probs = np.zeros((logits.size(0),), dtype=np.float32)
    max_tokens= np.zeros((logits.size(0),), dtype=np.float32)
    for i in range(logits.size(0)):
        probi=torch.nn.functional.log_softmax(
            logits[i, prefix_lens[i] - 1, :], dim=0
        )
        a_probs[i]=probi[a_tok].item()
        b_probs[i]=probi[b_tok].item()
        max_tokens[i]=probi.argmax().item()
    correct_probs=[]
    correct_max=[]
    for i in range(len(which_correct)):
        label=which_correct[i]
        if label==0:                   
            correct_probs.append(a_probs[i]>=b_probs[i])
            correct_max.append(max_tokens[i]==a_tok[0])
        else:                  
            correct_probs.append(a_probs[i]<b_probs[i])
            correct_max.append(max_tokens[i]==b_tok[0])
    ret={key+"_probs":correct_probs,key+"_correct":correct_max}
    metrics=replace_tf_with_detailed_acc(ret)
    return metrics
def lw_eval(record,cfg,model,tok):
    metrics={}
    tf_prompts=record["efficacy_evaluation"][:2]
    metrics_tf=test_true_false(cfg,model,tok,tf_prompts,[1,0],key="rewrite_tf")
    metrics=metrics|metrics_tf
    ab_prompts=[record["answer_a"],record["answer_b"]]
    metrics_ab=test_a_b(cfg,model,tok,ab_prompts,which_correct=[0,1],
                 key="rewrite_ab",question=record["question"],statement=False)
    metrics=metrics|metrics_ab
    abs_prompts=[tf_prompts[1],tf_prompts[0]]
    metrics_abs=test_a_b(cfg,model,tok,abs_prompts,which_correct=[0,1],
                 key="rewrite_abs",question=record["question"],statement=True)
    metrics=metrics|metrics_abs
    metrics_random=lw_eval_random(record, cfg, model, tok)
    metrics=metrics|metrics_random
    return metrics
def lw_eval_random(record,cfg,model,tok):
    metrics={}
    eff=record["efficacy_evaluation"][:2]
    tf_prompts=[eff[0],eff[1].replace(record["target_new"],record["target_random"])]
    metrics_tf=test_true_false(cfg,model,tok,tf_prompts,[1,0],key="rewrite_rtf")
    metrics=metrics|metrics_tf
    ab_prompts=[record["target_random"],record["answer_b"]]
    metrics_ab=test_a_b(cfg,model,tok,ab_prompts,which_correct=[0,1],
                 key="rewrite_rab",question=record["question"],statement=False)
    metrics=metrics|metrics_ab
    abs_prompts=[tf_prompts[1],tf_prompts[0]]
    metrics_abs=test_a_b(cfg,model,tok,abs_prompts,which_correct=[0,1],
                 key="rewrite_rabs",question=record["question"],statement=True)
    metrics=metrics|metrics_abs
    return metrics
