from PIL import Image, ImageDraw
import math
import random
import numpy as np
import matplotlib.pyplot as plt

random_seed = 123456789
random.seed(random_seed)

def to_rgb(image):
	rgb = image.convert("RGB")
	width = image.width
	height = image.height
	matrix = np.zeros((width, height, 3))
	for x in range(width):
		for y in range(height):
			matrix[x, y, :] = rgb.getpixel((x, y))
	return matrix

def to_gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def add_noise(matrix, noise_level):
	noise = np.random.randn(matrix.shape[0], matrix.shape[1]) * noise_level
	matrix = matrix + noise
	matrix[matrix > 255] = 255
	matrix[matrix < 0] = 0
	return matrix

def to_binary(matrix, threshold):
	result = np.array(matrix)
	result[matrix > threshold] = 1
	result[matrix <= threshold] = -1
	return result

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def gibbs_sampling(input_, num_iters = 5, J = 1):
	DX = [-1, 1, 0, 0]
	DY = [0, 0, -1, 1]
	X = np.zeros(input_.shape)
	X[:, :] = input_[:, :]
	width = X.shape[0]
	height = X.shape[1]
	for iter in range(num_iters):
		X_next = np.zeros(X.shape)
		for x in range(width):
			for y in range(height):
				agree = 0
				disagree = 0
				for i in range(len(DX)):
					x_ = x + DX[i]
					y_ = y + DY[i]
					if x_ >= 0 and x_ < width and y_ >= 0 and y_ < height:
						if X[x_][y_] == X[x][y]:
							agree += 1
						else:
							disagree += 1
				eta = X[x][y] * (agree - disagree)
				prob = sigmoid(2 * J * eta)
				sample = np.random.rand()
				if sample <= prob:
					X_next[x][y] = 1
				else:
					X_next[x][y] = -1
		X = X_next
	return X

image = Image.open("jim_simons.jpg").convert('LA')
matrix = to_gray(to_rgb(image))
noised_matrix = add_noise(matrix, noise_level = 25)

X = to_binary(noised_matrix, threshold = 50)
X_gibbs = gibbs_sampling(X, num_iters = 5, J = 1)

plt.subplot(1, 3, 1)
plt.imshow(matrix.transpose())
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(X.transpose() * 255)
plt.title("White noised")

plt.subplot(1, 3, 3)
plt.imshow(X_gibbs.transpose() * 255)
plt.title("Gibbs sampling on 2D Ising model")
plt.show()


gibbs_1 = gibbs_sampling(X, num_iters = 1, J = 1)
gibbs_3 = gibbs_sampling(X, num_iters = 3, J = 1)
gibbs_5 = gibbs_sampling(X, num_iters = 5, J = 1)

plt.subplot(1, 4, 1)
plt.imshow(X.transpose() * 255)
plt.title("White noised")

plt.subplot(1, 4, 2)
plt.imshow(gibbs_1.transpose() * 255)
plt.title('Gibbs sampling (1 iteration)')

plt.subplot(1, 4, 3)
plt.imshow(gibbs_3.transpose() * 255)
plt.title("Gibbs sampling (3 iterations)")

plt.subplot(1, 4, 4)
plt.imshow(gibbs_5.transpose() * 255)
plt.title("Gibbs sampling (5 iterations)")
plt.show()
