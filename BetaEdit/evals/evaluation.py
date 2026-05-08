from evals.counterfact import eval_counterfact
from evals.zsre import eval_zsre
from evals.wiki_cf import eval_wiki_cf
from evals.mquake_cf import eval_mquake_cf
from evals.glue_eval.glue_eval import GLUEEval
import numpy as np
def eval_one_edit(cfg,model,tok,edit):
    metrics=None
    data_name=cfg.data
    if data_name=="multi_counterfact_20877":
        metrics=eval_counterfact(cfg,model,tok,edit)
    elif data_name=="zsre_mend_eval_19086":
        metrics=eval_zsre(cfg,model,tok,edit)
    elif data_name=="mquake_cf_9218":
        metrics=eval_mquake_cf(cfg,model,tok,edit)
    elif data_name=="wiki_cf_2266":
        metrics=eval_wiki_cf(cfg,model,tok,edit)
    elif data_name=="zsre":
        metrics=eval_zsre(cfg,model,tok,edit)
    else:
        raise ValueError("dataset {} not recognized".format(data_name))
    return metrics
def eval_algo(cfg, model, tok, data):
    all_metrics = {}
    for edit in data:                      
        metrics = eval_one_edit(cfg, model, tok, edit)
        if len(all_metrics) == 0:
            for key, value in metrics.items():
                all_metrics[key] = [value]
        else:
            for key, value in metrics.items():
                all_metrics[key].append(value)
    avg_metrics = {}
    for key, value in all_metrics.items():
        avg_metrics[key] = np.round(np.mean(value), 3).item()
    if cfg.glue_eval:
        glue_results = {}
        shots=0
        glue_eval = GLUEEval(cfg,model, tok, number_of_tests=200,nli_number_of_few_shots=shots,dialogue_number_of_few_shots=shots,
                             sentiment_analysis_number_of_few_shots=shots,sst_number_of_few_shots=shots,mrpc_number_of_few_shots=shots,
                             cola_number_of_few_shots=shots,rte_number_of_few_shots=shots,mmlu_number_of_few_shots=shots
                             )
        out_file = "glue_generation.json"
        glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag=True, sst_flag=True, cola_flag=True,
                                          rte_flag=True, mmlu_flag=True, mrpc_flag=True,sentiment_analysis_flag=True,
                                          dialogue_flag=True)
        avg_metrics = avg_metrics | glue_results
    return avg_metrics
