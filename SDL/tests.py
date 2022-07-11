from als import ALS
import numpy as np

W0 = np.random.rand(10,5)
H0 = np.random.rand(5,20)
X = W0 @ H0
rank = 5

model = ALS(X, rank, W0, H0, non_negativity_W=True, non_negativity_H=True)
W, H = model.fit()

print(f"Reconstruction Error: {np.linalg.norm(X-W@H)**2}")
print(f"Dicitonary Error: {np.linalg.norm(W0 - W)**2}")
print(f"Encoding Error: {np.linalg.norm(H0-H)**2}")