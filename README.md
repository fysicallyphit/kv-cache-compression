## Compress KV cache during inference by using Single Value Decomposition to approximate K and V with lower rank matrices.

### Problem: 
Inference cost in transformers models O(n^2). KV cache compression reduces this quadratic growth -> linear, but not all dimensions in the stored KV matrices are equally significant. We can prune the insignificant directions to reduce storage. 

### Idea: 
Single Value Decomposition takes in a matrix M and returns Matrices U, S, and V_t, such that 
```
M = U @ diag(S) @ V
```
Instead of representing K and V as matrices of (seq_len, head_dim), we can take the top r basis vectors and represent K and V as linear combinations of those bases. 

S, the rectangular diagonal matrix, gives singular values in decreasing order. Plot S to see the decay.

## Decay of Singular Values
<img width="634" height="476" alt="Screenshot 2026-03-18 at 6 43 39 PM" src="https://github.com/user-attachments/assets/1ea24226-a0e1-4e77-9e72-cc029bb960d0" />

Exponential decay = we can prune the lowest and insignificant singular values.

Linear decay = all singular values are significant and likely linearly independent.

## Error vs. Memory:
<img width="635" height="474" alt="Screenshot 2026-03-14 at 4 27 38 PM" src="https://github.com/user-attachments/assets/3ee02b2e-8218-4862-a56a-c4e27a11906b" />

As expected, error increases as we use less storage for singular value approximation.

## Compressibility by Layer:
<img width="625" height="472" alt="Screenshot 2026-03-15 at 3 33 22 PM" src="https://github.com/user-attachments/assets/9baa7137-03cd-47ac-bd99-a79c78659903" />

Not all layers are uniformly compressible. Ex: Layers 8 and 12 are near full-rank.

## Adapted Rank Compression:
<img width="635" height="471" alt="Screenshot 2026-03-18 at 5 39 01 PM" src="https://github.com/user-attachments/assets/18c12a3b-eb10-4870-beca-c99373457bae" />


Metrics for success: memory and latency savings, quality retention.

All code is mine

Model structure:
```
BertModel( (embeddings): BertEmbeddings( (word_embeddings): Embedding(30522, 768, padding_idx=0) (position_embeddings): Embedding(512, 768) (token_type_embeddings): Embedding(2, 768) (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True) (dropout): Dropout(p=0.1, inplace=False) )

(encoder): BertEncoder( (layer): ModuleList( (0-11): 12 x BertLayer( (attention): BertAttention( (self): BertSelfAttention( (query): Linear(in_features=768, out_features=768, bias=True) (key): Linear(in_features=768, out_features=768, bias=True) (value): Linear(in_features=768, out_features=768, bias=True) (dropout): Dropout(p=0.1, inplace=False) ) (output): BertSelfOutput( (dense): Linear(in_features=768, out_features=768, bias=True) (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True) (dropout): Dropout(p=0.1, inplace=False) ) ) (intermediate): BertIntermediate( (dense): Linear(in_features=768, out_features=3072, bias=True) (intermediate_act_fn): GELUActivation() ) (output): BertOutput( (dense): Linear(in_features=3072, out_features=768, bias=True) (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True) (dropout): Dropout(p=0.1, inplace=False) ) ) ) ) (pooler): BertPooler( (dense): Linear(in_features=768, out_features=768, bias=True) (activation): Tanh() ) )
```
