import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, plot_3d_scatter
from gen_data import normalise_data

# computes the entropy of a tensor
def calc_entropy(x:torch.Tensor):
    '''
    Computes the (Shannon) entropy of a tensor based on its empirical distribution
    '''
    x = x.flatten()
    freq = x.unique(return_counts=True)[1]
    probs = freq/torch.sum(freq)
    return -torch.multiply(probs, torch.log(probs)).sum()

X, Y = load_data('./data/','train_complex_temp.pkl')

# principal component analysis of output
# standardise data
Z_scored_Y = normalise_data(Y.view(-1,16)) # normalise data
Z_scored_Y = Z_scored_Y.cpu().numpy()

cov = np.cov(Z_scored_Y.T)

# eigendecomposition and sort eigenvalues
eigenvals, eigenvecs = np.linalg.eig(cov)

idx = eigenvals.argsort()[::-1]   
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:,idx].T

pca = Z_scored_Y @ eigenvecs[:3].T

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(projection='3d')
ax.scatter(pca[:,0], pca[:,1], pca[:,2], s=0.5)
ax.set_xlabel('Principle component 1')
ax.set_ylabel('Principle component 2')
ax.set_zlabel('Principle component 3')
plt.show()