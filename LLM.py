# CHAPTER 2: TOKENIZING TEXT
# 2.1 and 2.2 are just setups

import os
import urllib.request
import re

if not os.path.exists("the-verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    urllib.request.urlretrieve(url, "the-verdict.txt")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

text = "Hello, world. This, is a test."
result = re.split(r'([,.]|\s)', text)
result = [item for item in result if item.strip()]

# Strip whitespace from each item and then filter out any empty strings.

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
preprocessed = result
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# 2.3 CONVERTING TOKENS INTO TOKEN IDs

all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)

vocab = {token:integer for integer, token in enumerate(all_words)}

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab #A
        self.int_to_str = {i:s for s,i in vocab.items()} #B

    def encode(self, text): #C
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids): #D
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuation.
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #E
        return text

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)

tokenizer.decode(ids)

# 2.4 ADDING SPECIAL CONTEXT TOKENS

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab #A
        self.int_to_str = {i:s for s,i in vocab.items()} #B

    def encode(self, text): #C
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int #A
            else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids): #D
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuation.
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #E
        return text
    
tokenizer = SimpleTokenizerV2(vocab)
text = """Hello, do you like tea? Is this bhenchod?"""
tokenizer.encode(text)
tokenizer.decode(tokenizer.encode(text))

# 2.5 BYTE PAIR ENCODING

import tiktoken
from importlib.metadata import version

tokenizer = tiktoken.get_encoding("gpt2")
tokenizer.encode("Hello world")
tokenizer.decode(tokenizer.encode("Hello World"))

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknwonPlace"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# 2.6 DATA SAMPLING WITH A SLIDING WINDOW

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader    

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
print(raw_text)
data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
