import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

K_matrices = []

def hook_fn(module, input, output):
    K_matrices.append(output)    

model.encoder.layer[0].attention.self.key.register_forward_hook(hook_fn)

inputs = tokenizer("the quick brown fox jumped over the lazy dog" * 20,  return_tensors ="pt")
K_matrices.clear() 
outputs = model(**inputs)
seq_len = inputs["input_ids"].shape[1]
print(seq_len)

K = K_matrices[0]
K = K.reshape(1, seq_len, 12, 64).transpose(1,2)
K_head = K[0,0,:,:].detach().numpy()

U, S, Vt = np.linalg.svd(K_head, full_matrices = False)
#plt.plot(S)
#plt.show() # see how fast S decays

for r in [1, 2, 4, 8, 16, 32, 64]:
    K_new = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
    error = np.linalg.norm(K_head - K_new)
    print(r, error)

