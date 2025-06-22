import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import tiktoken
from dataloader import GPTDatasetV1, create_dataloader
from multihead_attention import MultiHeadAttentionWrapper, MultiHeadAttention


with open("small-text-sample.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer.encode(raw_text)

vocab_size = 50257
output_dim = 256
max_len = 1024
context_length = max_len

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length)


for batch in dataloader:
    x, y = batch

    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    input_embeddings = token_embeddings + pos_embeddings

    break

print(input_embeddings.shape)


torch.manual_seed(123)

context_length = max_length
d_in = output_dim

num_heads=2

#d_out = d_in // num_heads
#mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads)

d_out = d_in
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)


batch = input_embeddings
context_vecs = mha(batch)

print("context_vecs.shape:", context_vecs.shape)

