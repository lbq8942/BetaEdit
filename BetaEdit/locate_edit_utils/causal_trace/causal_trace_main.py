import argparse
import json
import os
import re
from collections import defaultdict
import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from locate_edit_utils.causal_trace.knowns import KnownsDataset
from locate_edit_utils.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.runningstats import Covariance, tally
DATA_DIR="/home/liubingqing/project/MI/data"
def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)
    aa(
        "--model_name",
        default="gpt2-xl",
    )
    aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    aa("--gpu", default=0, type=int)
    aa("--layers_select", default=0, type=int)
    args = parser.parse_args()
    modeldir = f'r{args.replace}_{args.model_name.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    torch_dtype = torch.float16 if "20b" in args.model_name else None
    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype,gpu=args.gpu)
    if args.fact_file is None:
        knowns = KnownsDataset(DATA_DIR)               
    else:
        with open(args.fact_file) as f:
            knowns = json.load(f)
    knowns.data=knowns.data[:2]                               
    noise_level = args.noise_level              
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, [k["subject"] for k in knowns]
            )                     
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            noise_level = collect_embedding_gaussian(mt)
            print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])
    for knowledge in tqdm(knowns):         
        known_id = knowledge["known_id"]
        if args.layers_select ==0:
            kind_ttype=[None, "mlp", "attn"]
        else:
            kind_ttype=[None]
        for kind in kind_ttype:
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            if not os.path.isfile(filename):                                                       
                result = calculate_hidden_flow(
                    mt,        
                    knowledge["prompt"],      
                    knowledge["subject"],     
                    expect=knowledge["attribute"],                
                    kind=kind,
                    noise=noise_level,             
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            if not numpy_result["correct_prediction"]:                                       
                tqdm.write(f"Skipping {knowledge['prompt']}")
                continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            pdfname = f'{pdf_dir}/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}.pdf'
            if known_id > 200:
                continue
            plot_trace_heatmap(plot_result, savepdf=False)                
            if kind==None:
                scores = plot_result["scores"]
                last_id = plot_result["subject_range"][-1] - 1
                print("The recommended layer to be edited (in descending order):")
                print(scores[last_id, :].argsort()[::-1])
def trace_with_patch(
    mt,             
    inp,                   
    states_to_patch,                                                         
    answers_t,                                   
    tokens_to_mix,                                           
    noise=0.1,                         
    uniform_noise=False,
    replace=False,                                             
    trace_layers=None,                                    
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.
    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.
    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """
    model=mt.model
    rs = numpy.random.RandomState(1)                                               
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)               
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:                                                        
        patch_spec[l].append(t)                                                                     
    embed_layername = layername(mt.config, 0, "embed")                                                                             
    def untuple(x):
        return x[0] if isinstance(x, tuple) else x
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise
    def patch_rep(x, layer):
        if layer == embed_layername:                              
            if tokens_to_mix is not None:                 
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))                                              
                ).to(x.device)                
                if replace:                                             
                    x[1:, b:e] = noise_data                               
                else:
                    x[1:, b:e] += noise_data                       
            return x
        if layer not in patch_spec:
            return x
        h = untuple(x)                   
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]                      
        return x
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,                         
        edit_output=patch_rep,            
    ) as td:                                                  
        outputs_exp = model(**inp)                                   
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced
    return probs
def trace_with_repatch(
    mt,             
    inp,                   
    states_to_patch,                                                         
    states_to_unpatch,                                                              
    answers_t,                                   
    tokens_to_mix,                                           
    noise=0.1,                         
    uniform_noise=False,
):
    model=mt.model
    rs = numpy.random.RandomState(1)                                               
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)
    embed_layername = layername(mt.config, 0, "embed")
    def untuple(x):
        return x[0] if isinstance(x, tuple) else x
    def patch_rep(x, layer):
        if layer == embed_layername:
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    return probs
def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1),device=mt.model.device)                                                    
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])                               
    if expect is not None and answer.strip() != expect:                      
        return dict(correct_prediction=False)
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)                      
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    low_score = trace_with_patch(
        mt, inp, [], answer_t, e_range, noise=noise, uniform_noise=uniform_noise
    ).item()                                          
    if not kind:
        differences = trace_important_states(                     
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:                            
        differences = trace_important_window(
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )
    differences = differences.detach().cpu()                                    
    return dict(
        scores=differences,           
        low_score=low_score,         
        high_score=base_score,           
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )
def trace_important_states(
    mt,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    model=mt.model
    ntoks = inp["input_ids"].shape[1]                     
    table = []
    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(        
                mt,
                inp,
                [(tnum, layername(mt.config, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)
def trace_important_window(
    mt,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):                                                  
    model=mt.model
    config=mt.config
    ntoks = inp["input_ids"].shape[1]         
    table = []
    if token_range is None:
        token_range = range(ntoks)             
    for tnum in token_range:
        row = []
        for layer in range(num_layers):              
            layerlist = [
                (tnum, layername(config, L, kind))                     
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))                        
                )                                                                                
            ]
            r = trace_with_patch(
                mt,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )                                        
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)
class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """
    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
            gpu=0,
    ):
        from main import MODEL_PATH
        import yaml
        with open('../../configs/llms/{}.yaml'.format(model_name), 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)                      
        model_name=self.config["name"]
        model_name_or_path = MODEL_PATH.get(model_name, model_name)
        self.device=torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            ).to(self.device)
            nethook.set_requires_grad(False, model)
            model.eval()                        
        self.tokenizer = tokenizer
        self.model = model
        self.num_layers =self.model.config.num_hidden_layers
    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )
def layername(config, num, kind=None):
    if kind=="embed":
        return config["emb_module"]
    elif kind=="mlp":
        return config["mlp_module_tmp"].format(num)
    elif kind=="attn":
        return config["attn_module_tmp"].format(num)
    elif kind==None:
        return config["layer_module_tmp"].format(num)
def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()
def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap(result, savepdf)
def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)                      
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):                          
        labels[i] = labels[i] + "*"                       
    with plt.rc_context(rc={"font.family": "DejaVu Serif"}):             
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],                                              
            vmin=low_score,                         
        )
        ax.invert_yaxis()                                  
        ax.set_yticks([0.5 + i for i in range(len(differences))])            
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)]) 
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)         
        if not modelname:
            modelname = "GPT"                
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")                                      
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)           
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:                   
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)                   
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
def plot_all_flow(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind)
def make_inputs(tokenizer, prompts, device="cpu"):                  
    token_lists = [tokenizer.encode(p) for p in prompts]                                                
    maxlen = max(len(t) for t in token_lists)                         
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]                             
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]             
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )
def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]
def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)                                                                  
    whole_string = "".join(toks)                          
    char_loc = whole_string.index(substring)           
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)                                                                               
def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts,mt.model.device)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result
def predict_from_input(model, inp):
    out = model(**inp)["logits"]                                                                                                                        
    probs = torch.softmax(out[:, -1], dim=1)                               
    p, preds = torch.max(probs, dim=1)                                   
    return preds, p         
def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:       
        inp = make_inputs(mt.tokenizer, [s],device=mt.model.device)                         
        with nethook.Trace(mt.model, layername(mt.config, 0, "embed")) as t:
            mt.model(**inp)                                                        
            alldata.append(t.output[0])              
    alldata = torch.cat(alldata)                                   
    noise_level = alldata.std().item()
    return noise_level                                             
def get_embedding_cov(mt):
    model = mt.model
    tokenizer = mt.tokenizer
    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = 100                                             
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)
    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename = None
    batch_tokens = 100
    progress = lambda x, **k: x
    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, torch.device("cpu"))
                del batch["position_ids"]
                with nethook.Trace(model, layername(mt.config, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()
def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer
def collect_embedding_gaussian(mt):
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)
def collect_embedding_tdist(mt, degree=3):
    u_sample = torch.from_numpy(
        numpy.random.RandomState(2).chisquare(df=degree, size=1000)
    )
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)
    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        student = factor * gauss
        return student
    return normal_to_student
if __name__ == "__main__":
    main()
