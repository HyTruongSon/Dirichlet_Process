import numpy as np 
import random
import sklearn
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


seed = 12345678
np.random.seed(seed = seed)


def sample_index(prob):
	index = np.argsort(- prob)
	num_components = prob.shape[0]
	x = random.uniform(0, 1) * np.sum(prob)
	s = 0
	select = -1
	for j in range(num_components):
		s += prob[index[j]]
		if s >= x:
			select = index[j]
			break
	return select


def generate_gaussian_mixtures(num_dim, num_components, num_data):
	# Generate mixture components
	mean = []
	cov = []
	for i in range(num_components):
		m = np.random.rand(num_dim)
		mean.append(m)
		c = np.random.rand(num_dim, num_dim)
		c = np.dot(c, c.transpose())
		
		# Test: Just identity
		# c = np.eye(num_dim)
		
		cov.append(c)
	pi = np.random.rand(num_components)
	pi = pi - np.min(pi) + 1
	pi /= np.sum(pi)
	pi = - np.sort(- pi)

	# Generate data points
	n = np.zeros(num_components)
	for i in range(num_data):
		select = sample_index(pi)
		n[select] += 1

	for i in range(num_components):
		d = np.random.multivariate_normal(mean[i], cov[i], int(n[i]))
		if i == 0:
			data = d
		else:
			data = np.concatenate((data, d), axis = 0)

	return pi, mean, cov, np.array(data)


def stick_breaking(alpha, epsilon):
	beta = []
	pi = []
	total = 0.0
	while True:
		b = np.random.beta(1, alpha)
		beta.append(b)
		p = b * (1 - total)
		pi.append(p)
		total += p
		if 1 - total < epsilon:
			break
	return np.array(pi), np.array(beta)


def fit_dp(data, alpha, epsilon, m0, V0, S0, nu0, num_iters):
	# Step 1
	pi, beta = stick_breaking(alpha, epsilon)
	K = pi.shape[0]

	print("Initial number of components:", K)

	num_data = data.shape[0]
	num_dim = data.shape[1]

	mu = []
	sigma = []
	for k in range(K):
		m = np.random.multivariate_normal(m0, V0)
		mu.append(m)
		
		# There is no option for inverse Wishart distribution for now
		# Choose the mode of inverse Wishart instead
		mode = S0 / (nu0 + num_dim + 1)
		sigma.append(mode)

	
	for iter in range(num_iters):
		# Step 2
		prob = np.zeros((num_data, K))
		for k in range(K):
			y = multivariate_normal.pdf(data, mean = mu[k], cov = sigma[k])
			prob[:, k] = pi[k] * y

		z = np.zeros(num_data)
		N = np.zeros(K)

		for i in range(num_data):
			select = sample_index(prob[i, :].transpose())
			z[i] = select
			N[select] += 1

		for k in range(K):
			if N[k] > 0:
				# Step 3
				xk = data[z == k, :]
				xk_bar = np.sum(xk, axis = 0) / N[k]
				V0_inverse = np.linalg.inv(V0)
				sigmak_inverse = np.linalg.inv(sigma[k])
				Vk_inverse = V0_inverse + N[k] * sigmak_inverse
				Vk = np.linalg.inv(Vk_inverse)
				mk = np.matmul(Vk, N[k] * np.matmul(sigmak_inverse, xk_bar) + np.matmul(V0_inverse, m0))
				mu[k] = np.random.multivariate_normal(mk, Vk)

				# Step 4
				xk = xk - mu[k]
				Sk = S0 + np.matmul(xk.transpose(), xk)
				nuk = nu0 + N[k]

				# There is no option for inverse Wishart distribution for now
				# Choose the mode of inverse Wishart instead
				mode = Sk / (nuk + num_dim + 1)
				sigma[k] = mode

		# Finally shrink all components with no assignment
		pi_new = []
		mu_new = []
		sigma_new = []
		for k in range(K):
			if N[k] > 0:
				pi_new.append(pi[k])
				mu_new.append(mu[k])
				sigma_new.append(sigma[k])
		pi = np.array(pi_new)
		pi /= np.sum(pi)
		mu = mu_new
		sigma = sigma_new
		K = len(mu)

	return pi, mu, sigma


num_dim = 2
num_components = 3
num_data = 1000
pi, mean, cov, data = generate_gaussian_mixtures(num_dim, num_components, num_data)


alpha = 1
epsilon = 0.05
m0 = np.zeros(num_dim)
V0 = np.eye(num_dim)
S0 = np.eye(num_dim)
nu0 = 3
num_iters = 100
pi_, mean_, cov_ = fit_dp(data, alpha, epsilon, m0, V0, S0, nu0, num_iters)

plt.plot(data[:, 0], data[:, 1], 'r.')
K = len(mean)
for k in range(K):
	plt.plot(mean[k][0], mean[k][1], 'bx')
plt.title("Ground-truth means")
plt.show()

plt.plot(data[:, 0], data[:, 1], 'r.')
K = len(mean_)
print("Final number of components:", K)
for k in range(K):
	plt.plot(mean_[k][0], mean_[k][1], 'bx')
plt.title("Dirichlet Process Gaussian Mixture Model -- Gibbs sampling " + str(num_iters) + " iterations")
plt.show()


def density(x, y, pi, mean, cov):
	K = pi.shape[0]
	f = np.zeros(x.shape)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			sample = np.array([x[i][j], y[i][j]])
			total = 0
			for k in range(K):
				total += pi[k] * multivariate_normal.pdf(sample, mean = mean[k], cov = cov[k])
			f[i][j] = total
	return f


xmin = np.min(data[:, 0])
xmax = np.max(data[:, 0])
ymin = np.min(data[:, 1])
ymax = np.max(data[:, 1])
x = np.linspace(start = xmin, stop = xmax, num = 10)
y = np.linspace(start = ymin, stop = ymax, num = 10)
X, Y = np.meshgrid(x, y)

Z = np.log(density(X, Y, pi, mean, cov))
plt.contour(X, Y, Z, colors = 'black')
plt.title('Ground-truth density')
plt.show()

Z = np.log(density(X, Y, pi_, mean_, cov_))
plt.contour(X, Y, Z, colors = 'black')
plt.title('Dirichlet Process density')
plt.show()