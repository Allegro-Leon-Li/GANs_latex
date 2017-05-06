from numpy import *
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata


mnist = fetch_mldata("MNIST original")
mnist=mnist.data[:70000]
pca = PCA(n_components=100)
mnist_mean=mnist.mean(axis=0)
# plt.imshow(mnist_mean.reshape(28,28),cmap='Greys_r')
# plt.show()
# exit()
mnist_demeaned=mnist-mnist_mean
cov=np.cov(np.transpose(mnist_demeaned))
ev,_=np.linalg.eig(cov)
# print(ev[700:705])
# print(np.sum(ev[99:784]))
# plt.plot(ev)
# plt.show()
# exit()
mnist_pca=pca.fit_transform(mnist_demeaned)


random_test=np.random.uniform(-1., 100., size=(200,100))
test_pca_inverse=pca.inverse_transform(random_test)
print("reconstruct: ", test_pca_inverse.shape)
test_after=test_pca_inverse+mnist_mean
test_after = (test_after+255.)/2

h,w = 28, 28
img = np.zeros((h * 14, w * 14))
fig = plt.figure()
for i in range(196):
    j=i%14
    i=i//14
    img[i*h:i*h+h, j*w:j*w+w] = test_after[i].reshape(28,28)
plt.imshow(img, cmap='Greys_r')
plt.savefig('pcaresult.png',bbox_inches='tight')
plt.close(fig)





