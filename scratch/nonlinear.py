import causaldag as cd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

d = cd.DAG(arcs={(0, 1)})
g = cd.rand.rand_nn_functions(d, num_layers=2)
samples = g.sample(1000)
iv_samples = g.sample_interventional({1: cd.ShiftIntervention(5)}, 1000)
plt.clf()
plt.ion()
plt.scatter(samples[:, 0], samples[:, 1], label='observational')
plt.scatter(iv_samples[:, 0], iv_samples[:, 1], label='interventional')
plt.show()
plt.legend()
print(samples.mean(axis=0))
print(np.corrcoef(samples, rowvar=False))
print(np.corrcoef(iv_samples, rowvar=False))
