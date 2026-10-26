import numpy as np

# 1. Load your .npy file
data = np.load('data/embeddings/kubernetes_minilm_embeddings.npy')

# 2. Tell NumPy never to truncate the data
np.set_printoptions(threshold=np.inf)

# 3. Print the entire array to your terminal
print(data)
