import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc

random_seed = 123456789
random.seed(random_seed)

alpha = 5
K = 50
num_samples = 3

samples = []
for sample in range(num_samples):
	beta = []
	pi = []
	total = 0.0
	for k in range(K):
		# beta.append(sc.beta(1, alpha))
		beta.append(np.random.beta(1, alpha))
		pi.append(beta[k] * (1 - total))
		total += pi[k]
	samples.append(pi)

for sample in range(num_samples):
	pi = np.array(samples[sample])
	plt.subplot(1, num_samples, sample + 1)
	plt.bar(np.arange(pi.shape[0]), pi)
	plt.title('alpha = ' + str(alpha) + ', sample ' + str(sample + 1))
plt.show()