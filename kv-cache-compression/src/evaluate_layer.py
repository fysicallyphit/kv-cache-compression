import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from extract import extract_qkv
from compress import compute_svd

outputs, Q_layers, K_layers, V_layers = extract_qkv()
svd_K, svd_V = compute_svd(K_layers, V_layers)

def compression_experiment(Q_layers, K_layers, V_layers, svd_K, svd_V, X = 15):
    attention_errors = []
    adapted_rs = []
    compressed_KV_memories = []
    full_KV_memories = []

    # GROUND TRUTH:
    d_k = 64
    r_candidates = [1,2,4,8,16,32,64]
    
    assert len(Q_layers) == len(K_layers) == len(V_layers) == len(svd_K) == len(svd_V)
    
    for i in range(len(Q_layers)):
        Uk, Sk, Vtk = svd_K[i]
        Uv, Sv, Vtv = svd_V[i]
        Q = Q_layers[i]
        K = K_layers[i]
        V = V_layers[i]
        
        attention = softmax((Q @ K.T)/np.sqrt((d_k)), axis = -1) @ V
        
        chosen_r = r_candidates[-1]
        chosen_err = None
        chosen_mem = None

        for r in r_candidates:
            K_approx = Uk[:,:r] @ np.diag(Sk[:r]) @ Vtk[:r, :]
            Uk_approx = (Uk[:,:r]) # coordinates of each token as a linear combination of basis vectors
            Vtk_approx = (Vtk[:r, :]) # r basis vectors 

            V_approx = Uv[:,:r] @ np.diag(Sv[:r]) @ Vtv[:r, :]
            Uv_approx = (Uv[:,:r]) 
            Vtv_approx = (Vtv[:r, :]) 

            attention_approx = softmax((Q @ K_approx.T)/np.sqrt(d_k), axis = -1) @ V_approx
            r_error = np.linalg.norm(attention - attention_approx) / np.linalg.norm(attention)
            memory = Uk_approx.nbytes + Sk[:r].nbytes + Vtk_approx.nbytes + Uv_approx.nbytes + Sv[:r].nbytes + Vtv_approx.nbytes
            
            if r_error< X/100:
                chosen_r = r
                chosen_err = r_error
                chosen_mem = memory
                break
        
        if chosen_err is None:
            chosen_err = r_error
            chosen_mem = memory

        full_KV_memory = K.nbytes + V.nbytes

        adapted_rs.append(chosen_r)
        attention_errors.append(chosen_err)
        compressed_KV_memories.append(chosen_mem)
        full_KV_memories.append(full_KV_memory)     

    return adapted_rs, attention_errors, compressed_KV_memories, full_KV_memories

adapted_rs, attention_errors, compressed_KV_memories, full_KV_memories = compression_experiment(
    Q_layers, K_layers, V_layers, svd_K, svd_V, X = 10
)

plt.plot(adapted_rs)
plt.xlabel("Layers")
plt.ylabel("Rank")
plt.title("Adaptive Rank per Layer")
plt.legend()
plt.show()

# plt.scatter(compressed_KV_memories, attention_errors, label="compressed")
# plt.scatter(full_KV_memories, [0]*len(full_KV_memories), label="full")
# plt.xlabel("Storage (bytes)")
# plt.ylabel("Relative Attention Error")
# plt.title("Adaptive Rank: Error vs Storage")
# plt.legend()
# plt.show()

# plt.plot(range(len(attention_errors)), attention_errors)
# plt.xlabel("Layer")
# plt.ylabel("Attention Error SSE")
# plt.title("Adapted Rank Compressibility by Layer")
# plt.show()

