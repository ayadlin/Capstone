import pandas as pd
import numpy as np
import random
from IPython.display import display
import matplotlib.pyplot as plt

from sklearn.decomposition import NMF

def fit_nmf(M,r):
    nmf = NMF(n_components=r)
    nmf.fit(M)
    W = nmf.transform(M)
    H = nmf.components_
    return nmf.reconstruction_err_

error = [fit_nmf(resist_network_matrix,i) for i in range(1,10)]
plt.plot(range(1,10), error)
plt.xticks(range(1, 10))
plt.xlabel('number of latent topics')
plt.ylabel('Reconstruction Error')

# Fit using 2 hidden concepts
nmf = NMF(n_components=2)
nmf.fit(resist_network_matrix)
W = nmf.transform(resist_network_matrix)
H = nmf.components_
print('RSS = %.2f' % nmf.reconstruction_err_)

W, H = (np.around(x,2) for x in (W,H))
W = pd.DataFrame(W,index=network_genes.values())
H = pd.DataFrame(H,columns=network_drugs.values())

display(W)
display(H)
