# -*- coding: utf-8 -*-
'''
Description: chatglm2_6b lora finetuned by all risk data(17w+), to predict test risk data(4w+)
Author: Ocean
Date: 2023-07-19 11:23:13
LastEditors: Ocean
LastEditTime: 2023-07-19 11:41:17
'''
# Start Coding...

import os
import json
import time
import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModel

def ct():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def read_json_file(file):
    with open(file, "r", encoding='utf-8') as r:
        response = r.read()
        response = response.replace('\n', '')
        response = response.replace('}{', '},{')
        response = "[" + response + "]"
        return json.loads(response)
    
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print(f"{ct()} start!\n")

base_model_path = "/data/ouxin/llm/models/glm2_6b/full_weights"
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True).cuda().half()
print(f"{ct()} base_model load successful!\n")

ft_result_path = r"/data/ouxin/llm/code/chatglm_et_0713/risk_all_17w_lora"
ft_model = PeftModel.from_pretrained(
    base_model,
    ft_result_path,
    torch_dtype=torch.float16,
).cuda().half()
print(f"{ct()} finetuned_model load successful!\n")

def ask_model(prompt,infer_model,infer_tokenizer):
    response, _ = infer_model.chat(infer_tokenizer, prompt, history=[], eos_token_id=2, pad_token_id=2)
    return response

test_data_path = "/root/autodl-tmp/data/compress_json/all_test_0616.json"
json_list = read_json_file(test_data_path)[0]
print(f"{ct()} test data load successful!\n")
print(f"test data example:\n{json_list[0]}\n")

print(f"{ct()} try to predict one test data\n")
template = """
问：请你扮演一位风险分析师，你的任务是找到文本内的公司和公司对应的风险。{}
答：
"""
# template test
j = json_list[-1]
input_len = len(j['input'])
print(f"\n{ct()} input length {input_len}\n")
if input_len < 1200:
    st = time.time()
    prompt = template.format(j['input'])
    ans = ask_model(prompt,ft_model,tokenizer)
    et = time.time()
    print({'input':j['input'],'label':j['output'],'pred':ans,'consume':f"{(et-st):.2f}s"})
    
# predict all test
logs = open("/data/ouxin/llm/predict/predict_logs_0719.txt","w",encoding='utf-8')
result = []
i, n =  0, len(json_list)
for j in json_list:
    input_len = len(j['input'])
    dic = {}
    st = time.time()
    prompt = template.format(j['input'][:4096])
    ans = ask_model(prompt,ft_model,tokenizer)
    et = time.time()
    print(f"{i}/{n} {ct()} input length {input_len}, labels: {j['output']}, preds: {ans}, consume: {(et-st):.2f}s\n",file=logs)
    dic['inputs'] = j['input']
    dic['labels'] = j['output']
    dic['preds'] = ans
    result.append(dic)
    i += 1
logs.close()

save_path = f'/data/ouxin/llm/predict/glm2_6b_ft_predict_risk_test_0719.json'
with open(save_path,"w",encoding="utf-8") as f:
    json.dump(result,f,indent=4,ensure_ascii=False)