### KV Cache Compression via Low-Rank SVD Approximation
Compresses the KV cache during transformer inference by approximating key and value matrices with lower-rank representations using Singular Value Decomposition (SVD). Applied to BERT-base and analyzed layer-by-layer for compressibility, rank adaptation, and quality retention.

## Problem
Inference cost in transformers models O(n^2). KV cache compression reduces this quadratic growth -> linear, but not all dimensions in the stored KV matrices are equally significant. We can prune the insignificant directions to reduce storage. 

## Approach
SVD factorizes a matrix M as:
```M = U @ diag(S) @ V_t```
where S contains singular values in decreasing order. Rather than storing K and V at full rank (seq_len, head_dim), we retain only the top r singular vectors, expressing each matrix as a linear combination of the most significant basis directions. This reduces storage from O(seq_len × head_dim) to O(seq_len × r + head_dim × r).
The choice of r is guided empirically by the decay profile of the singular value spectrum.

## Decay of Singular Values
<img width="634" height="476" alt="Screenshot 2026-03-18 at 6 43 39 PM" src="https://github.com/user-attachments/assets/1ea24226-a0e1-4e77-9e72-cc029bb960d0" />
The decay profile determines compressibility:

Exponential decay → energy is concentrated in a few directions; aggressive rank reduction is safe
Linear decay → singular values are roughly equally significant; the matrix is near full-rank and compression will incur higher error

## Error vs. Memory Trade-off
<img width="635" height="474" alt="Screenshot 2026-03-14 at 4 27 38 PM" src="https://github.com/user-attachments/assets/3ee02b2e-8218-4862-a56a-c4e27a11906b" />
As expected, error increases as we use less storage for singular value approximation.

Reconstruction error increases monotonically as rank r decreases, consistent with the **Eckart–Young theorem**. This curve defines the Pareto frontier between compression ratio and approximation quality, and can be used to select a rank budget given an acceptable error threshold.

<img width="330" height="330" alt="Pareto_Efficient_Frontier_1024x1024" src="https://github.com/user-attachments/assets/ff4796f1-7ae0-4bcb-b5c2-d1dd75781298" />
Pareto Frontier

## Compressibility by Layer
<img width="625" height="472" alt="Screenshot 2026-03-15 at 3 33 22 PM" src="https://github.com/user-attachments/assets/9baa7137-03cd-47ac-bd99-a79c78659903" />

Not all layers are uniformly compressible. Ex: Layers 8 and 12 are near full-rank. Applying the same compression ratio as other layers affects performance disproportionately. Therefore, we should *not* use uniform rank compression.

## Adaptive Rank 
<img width="635" height="471" alt="Screenshot 2026-03-18 at 5 39 01 PM" src="https://github.com/user-attachments/assets/18c12a3b-eb10-4870-beca-c99373457bae" />
Rank is adapted based on each layer's singular value decay profile rather than a fixed r for all layers. Layers with rapid decay receive smaller rank budgets; layers near full-rank are either given higher rank or ignored entirely.

## Evaluation metrics: 
KV cache memory reduction, attention output reconstruction error, latency.

Model
Experiments run on bert-base-uncased (12 layers, 12 heads, hidden dim 768).
Model structure:

```
BertModel( (embeddings): BertEmbeddings( (word_embeddings): Embedding(30522, 768, padding_idx=0) (position_embeddings): Embedding(512, 768) (token_type_embeddings): Embedding(2, 768) (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True) (dropout): Dropout(p=0.1, inplace=False) )

(encoder): BertEncoder( (layer): ModuleList( (0-11): 12 x BertLayer( (attention): BertAttention( (self): BertSelfAttention( (query): Linear(in_features=768, out_features=768, bias=True) (key): Linear(in_features=768, out_features=768, bias=True) (value): Linear(in_features=768, out_features=768, bias=True) (dropout): Dropout(p=0.1, inplace=False) ) (output): BertSelfOutput( (dense): Linear(in_features=768, out_features=768, bias=True) (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True) (dropout): Dropout(p=0.1, inplace=False) ) ) (intermediate): BertIntermediate( (dense): Linear(in_features=768, out_features=3072, bias=True) (intermediate_act_fn): GELUActivation() ) (output): BertOutput( (dense): Linear(in_features=3072, out_features=768, bias=True) (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True) (dropout): Dropout(p=0.1, inplace=False) ) ) ) ) (pooler): BertPooler( (dense): Linear(in_features=768, out_features=768, bias=True) (activation): Tanh() ) )
```

All code is mine.
