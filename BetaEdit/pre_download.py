from transformers import AutoModelForCausalLM, AutoTokenizer
ds_name="wikipedia"
from datasets import load_dataset                             
raw_ds = load_dataset(
        ds_name,
    dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name]
    )
llms=["gpt2-xl","EleutherAI/gpt-j-6B"]
for llm in llms:
    model = AutoModelForCausalLM.from_pretrained(
        llm,
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(llm)
from modelscope import snapshot_download
llamas=["meta-llama/Llama-3-8B-Instruct","meta-llama/Llama-3.1-8B-Instruct",
              "meta-llama/Llama-3.2-3B-Instruct"]
llamas_modelscope=["LLM-Research/Meta-Llama-3-8B-Instruct","LLM-Research/Meta-Llama-3.1-8B-Instruct",
                   "LLM-Research/Llama-3.2-3B-Instruct"]
for llama in llamas_modelscope:
    model_id=snapshot_download(llama,
                               cache_dir="/home/liubingqing/.cache/modelscope")                        
edited_facts=[{"question":"The mother tongue of Danielle Darrieux is","answer":"American English"},
              {"question":"Danielle Darrieux was born in","answer":"China"}]
questions=["What is the mother tongue of Danielle Darrieux?",
           "The mother tongue of Danielle Darrieux is"]
'''
你需要使用那篇论文的代码完成对上述2个问题的回答，写两个问题，是希望你的代码是批处理的，而不是for循环先回答第一个question，然后第二个。
我需要如下5个东西：
1.answers#有两个question，那么答案也应该有2个。
2.ori_dist,ini_dist,enc_dist,dec_dist#论文图4的那4个分布，另外，对于给定的那2个question，按理说都应该回答American English，
也就是说生成2个单词，那么ori_dist应该有两个分布，即ori_dist的shape是[2,vocab_size]，其他3个dist同理。
你需要写一个函数，返回上述5个东西。
def func(edited_facts,questions):
    *****************
    return answers,(ori_dist,ini_dist,enc_dist,dec_dist)
'''
