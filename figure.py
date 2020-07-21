import causaldag as cd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from numpy.ma import masked_array

random.seed(181)
np.random.seed(181)

nsamples = 10

g = cd.GaussDAG(nodes=[0, 1, 2], arcs={(0, 1): 1, (0, 2): 1}, variances=[1, .2, .2])
obs_samples = g.sample(nsamples)

iv1_samples = g.sample_interventional({1: cd.GaussIntervention(0, 1)}, nsamples=nsamples)
iv01_samples = g.sample_interventional({1: cd.GaussIntervention(0, 1), 0: cd.GaussIntervention(1, .1)}, nsamples=nsamples)

cmap = 'bwr'
plt.clf()
os.makedirs('figures/example_data/', exist_ok=True)
plt.imshow(obs_samples, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('figures/example_data/obs.png', transparent=True, bbox_inches='tight')

plt.clf()
plt.imshow(iv1_samples, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.savefig('figures/example_data/iv2.png', transparent=True, bbox_inches='tight')

plt.clf()
plt.imshow(iv01_samples, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.savefig('figures/example_data/iv12.png', transparent=True, bbox_inches='tight')

all_samples = np.vstack((obs_samples, iv01_samples))
iv_indicators = np.vstack((np.zeros(nsamples)[:, None], np.ones(nsamples)[:, None]))
all_samples_plus = np.hstack((all_samples, iv_indicators))

plt.clf()
sample_mask = np.zeros(all_samples_plus.shape, dtype=bool)
sample_mask[:, 3] = True
indicator_mask = np.zeros(all_samples_plus.shape, dtype=bool)
indicator_mask[:, :3] = True
plt.imshow(masked_array(all_samples_plus, sample_mask), cmap=cmap)
plt.imshow(masked_array(all_samples_plus, indicator_mask), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.savefig('figures/example_data/obs_and_iv12.png', transparent=True, bbox_inches='tight')

