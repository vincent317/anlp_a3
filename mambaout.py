import gzip
import json

def load_json_gz(filename):
    with gzip.open(filename, 'r') as f:
        i = 0
        ret = []
        for json_line in f:
            if i == 10000:
                return ret
            data = json.loads(json_line)
            text = data['text']
            if len(text) > 2000:
                ret.append(text)
                i += 1

# Load 10000 strings from C4 dataset: https://huggingface.co/datasets/allenai/c4/tree/main/en
strings = load_json_gz('c4-train.00000-of-01024.json.gz')
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

device = 'cuda:0'

def copy_task(batch_size=64, batches=10, model=None, tokenizer=None, token_max_len=25, shuffle=False):
    print("start copying task for ", token_max_len)
    string_idx = 0
    success_copies = 0
    for _ in range(batches):
        cur_batch = []
        for count in range(batch_size):
            cur_batch.append(strings[count + string_idx])
        input_ids = tokenizer(cur_batch, return_tensors="pt", truncation=True, max_length=token_max_len).to(device)["input_ids"]
        if shuffle:
            col_perm = torch.randperm(input_ids.size(1))
            input_ids = input_ids[:, col_perm]
        input_ids = torch.cat([input_ids, input_ids], dim=1)
        input_ids = torch.cat([input_ids, input_ids[:, 0:1]], dim=1)
        output_ids = model.generate(input_ids, max_new_tokens = token_max_len-1)
        for count in range(batch_size):
            gold_token_len = (input_ids.shape[1]-1) // 2
            if torch.equal(input_ids[count][:gold_token_len], output_ids[count][gold_token_len*2:]):
                success_copies += 1
        string_idx += batch_size
    return success_copies / (batch_size * batches)

mamba_14b_tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
mamba_14b_model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf")
mamba_14b_model.to(device)
import time 
result_14b_time = []
result_14b = []
for i in [25, 50, 100, 150, 200, 250]:
    start_time = time.time()
    ans = copy_task(batch_size = 8, model = mamba_14b_model, tokenizer = mamba_14b_tokenizer, token_max_len=i)
    end_time = time.time()
    result_14b_time.append(end_time - start_time)
    result_14b.append(ans)
    print(ans)
    print('the time spent is', end_time - start_time)

print("the result 1.4b time is ", result_14b_time)
print("the result 1.4b is ", result_14b)

result_14b_shuffle_time = []
result_14b_shuffle = []
for i in [25, 50, 100, 150, 200, 250]:
    start_time = time.time()
    ans = copy_task(batch_size = 8, model = mamba_14b_model, tokenizer = mamba_14b_tokenizer, token_max_len=i, shuffle=True)
    end_time = time.time()
    result_14b_shuffle_time.append(end_time - start_time)
    result_14b_shuffle.append(ans)
    print(ans)
    print('the time spent is', end_time - start_time)

print("the shuffle 14b results", result_14b_shuffle)
print("the shuffle 14b time", result_14b_shuffle_time)

import ast

name_phone_pairs = []
with open('./phonebook.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line[-1] == ',':
            line = line[:-1]
        pair = ast.literal_eval(line)
        name_phone_pairs.append((pair[0], pair[1]))

# We found the phone book experiment hard to reproduce as the author did not give the exact prompt in the paper. 
# In addition, the accuracy fluctutaed a lot with the prompt we used.

import random
def phone_book_task(batch_size=64, batches=10, book_size=20, model=None, tokenizer=None):
    book = ''
    success_lookups = 0
    for i in range(book_size):
        name = name_phone_pairs[i][0]
        phone = name_phone_pairs[i][1]
        book = book + name + ': ' + phone + '.\n'
    book += 'Extract the person\'s phone number in the phonebook above. For example:\nPerson: Liam\nNumber: 436-725-2906\nPerson: Olivia\nNumber: 192-311-5790\n\n'
    for _ in range(batches):
        cur_batch = []
        gold_num_tokens_batch = []
        max_num_tokens = -1
        for _ in range(batch_size):
            query_pair_idx = random.randint(2, book_size)
            query = book + 'Person: ' + name_phone_pairs[query_pair_idx][0] + '\nNumber:'
            gold_num_tokens = tokenizer(name_phone_pairs[query_pair_idx][1], return_tensors="pt", padding=True).to(device)["input_ids"]
            max_num_tokens = max(max_num_tokens, gold_num_tokens.shape[1])
            gold_num_tokens_batch.append(gold_num_tokens[0])
            cur_batch.append(query)
        input_ids = tokenizer(cur_batch, return_tensors="pt", padding=True).to(device)["input_ids"]
        output_ids = model.generate(input_ids, max_new_tokens = max_num_tokens)
        for count in range(batch_size):
            true_number = tokenizer.decode(gold_num_tokens_batch[count])
            output_answer = tokenizer.decode(output_ids[count])
            if output_answer.count(true_number) > 1:
                success_lookups += 1
    return success_lookups / (batch_size * batches)

phone_ans = []
phone_time = []
t1 = time.time()

phone_ans.append(phone_book_task(model=mamba_14b_model, tokenizer=mamba_14b_tokenizer))
t2 = time.time()
phone_ans.append(phone_book_task(model=mamba_14b_model, tokenizer=mamba_14b_tokenizer, book_size=40))
t3 = time.time()
phone_ans.append(phone_book_task(batch_size=32, batches=20, model=mamba_14b_model, tokenizer=mamba_14b_tokenizer, book_size=80))
t4 = time.time()

phone_time =[t2 - t1, t3 - t2, t4 - t3]

print("the 14b phone result", phone_ans)
print("the 14b phone time", phone_time)

