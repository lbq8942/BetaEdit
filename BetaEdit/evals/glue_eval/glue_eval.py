import sys
import json
from evals.glue_eval.sst_eval import SSTEval
from evals.glue_eval.mrpc_eval import MRPCEval
from evals.glue_eval.cola_eval import COLAEval
from evals.glue_eval.rte_eval import RTEEval
from evals.glue_eval.mmlu_eval import MMLUEval
from evals.glue_eval.sentiment_analysis_eval import SENTIMENT_ANALYSIS_Eval
from evals.glue_eval.dialogue_eval import DIALOGUE_Eval
from evals.glue_eval.nli_eval import NLIEval
from util.perplexity import perplexity
from datasets import load_dataset
class GLUEEval():
    def __init__(self, cfg,model, tokenizer, number_of_tests = None, sst_number_of_few_shots = 0, mrpc_number_of_few_shots = 0, cola_number_of_few_shots = 0, rte_number_of_few_shots = 0, mmlu_number_of_few_shots = 0, sentiment_analysis_number_of_few_shots = 0, nli_number_of_few_shots = 0, dialogue_number_of_few_shots = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.sst_eval = SSTEval(cfg,model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = sst_number_of_few_shots)
        self.mrpc_eval = MRPCEval(cfg,model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = mrpc_number_of_few_shots)
        self.cola_eval = COLAEval(cfg,model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = cola_number_of_few_shots)
        self.rte_eval = RTEEval(cfg,model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = rte_number_of_few_shots)
        self.mmlu_eval = MMLUEval(cfg,model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = mmlu_number_of_few_shots)
        self.sentiment_analysis_eval = SENTIMENT_ANALYSIS_Eval(cfg,model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = sentiment_analysis_number_of_few_shots)
        self.nli_eval = NLIEval(cfg,model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = nli_number_of_few_shots)
        self.dialogue_eval = DIALOGUE_Eval(cfg,model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = dialogue_number_of_few_shots)
    def _save_generations(self, record_path, generations, task):
        output_filename = record_path.replace('.json', '_' + task + '_gen.json')
        with open(output_filename, "w") as f:
            json.dump(generations, f, indent=4)
    def evaluate(self, glue_results, record_path, perplexity_flag = False, sst_flag = False, mmlu_flag = False, mrpc_flag = False, cola_flag = False, rte_flag = False, nli_flag = False, sentiment_analysis_flag = False, dialogue_flag = False, gen_len = 5):
        if perplexity_flag:        
            raw_ds = load_dataset(
                        "wikitext",
                        dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")["wikitext"],
                        )
            glue_results['perplexity'] = perplexity(self.model, self.tokenizer, " ".join(raw_ds["train"]['text'][:20]), max_input_length=100)
        def get_acc(generations):
            count=0
            for i in range(len(generations)):
                if generations[i]['correct_new']:
                    count += 1
            return count/len(generations)
        if sst_flag:
            result_dict, generations = self.sst_eval.evaluate(gen_len)
            glue_results['sst'] = get_acc(generations)
        if mmlu_flag:
            result_dict, generations = self.mmlu_eval.evaluate(gen_len)
            glue_results['mmlu'] = get_acc(generations)       
        if mrpc_flag:
            result_dict, generations = self.mrpc_eval.evaluate(gen_len)
            glue_results['mrpc'] = get_acc(generations)
        if cola_flag:
            result_dict, generations = self.cola_eval.evaluate(gen_len)
            glue_results['cola'] = get_acc(generations)
        if rte_flag:
            result_dict, generations = self.rte_eval.evaluate(gen_len)
            glue_results['rte'] = get_acc(generations)
        if sentiment_analysis_flag:
            result_dict, generations = self.sentiment_analysis_eval.evaluate(gen_len)
            glue_results['sentiment_analysis'] = get_acc(generations)
        if nli_flag:
            result_dict, generations = self.nli_eval.evaluate(gen_len)
            glue_results['nli'] = get_acc(generations)
        if dialogue_flag:
            result_dict, generations = self.dialogue_eval.evaluate(gen_len)
            glue_results['dialogue'] = get_acc(generations)
        return glue_results
