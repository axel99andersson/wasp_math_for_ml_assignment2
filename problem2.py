import numpy as np
import matplotlib.pyplot as plt

def metropolis_hastings_sampler(p, N, theta0, epsilon):
  chain = np.zeros((2,N))
  theta = theta0
  for i in range(N):
    y = np.random.normal(loc=theta, scale=epsilon)
    alpha = p(y) / p(theta)

    if np.random.rand() <= alpha:
      chain[:,i] = y
      theta = y
    else:
      chain[:,i] = theta

  return chain

def approx_p(x):
  return np.exp(-(1/2)*x.T @ x)

chain = metropolis_hastings_sampler(p=approx_p, N=5000, theta0=7.0*np.ones(2), epsilon=0.5)
fig, axs = plt.subplots(2,2, figsize=(13,13))
axs[0,0].hist(chain[0,:], bins='auto')
axs[0,0].set_title("$x_1$")

axs[0,1].hist(chain[1,:], bins='auto')
axs[0,1].set_title("$x_2$")

axs[1,0].hist(chain[0,:50], bins='auto')
axs[1,0].set_title("$x_1$, First 50")

axs[1,1].hist(chain[1,:50], bins='auto')
axs[1,1].set_title("$x_2$, First 50")

fig.suptitle("Metropolis-Hastings Sampler: $\\theta_0=(7,7)$, $\\epsilon=0.5$")
plt.show()