import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
import jax.numpy as jnp
from jax import grad,vmap,jit
import jax
import matplotlib.pyplot as plt

#PROGRAM TIME
key = jax.random.key(0)
x_batch = jnp.linspace(-1, 1, 100).reshape((100, 1))
y_label_batch = 3 * x_batch ** 3 - x_batch ** 2 - 3 * x_batch + 2 + 0.2 * jax.random.normal(key, (100, 1))

plt.scatter(x_batch, y_label_batch, label="training data")
plt.legend()
plt.xlabel("x");
plt.ylabel("y")
plt.show()

def forward(theta,x):
    y = theta[0] + theta[1] * x + theta[2] * x ** 2 + theta[3] * x ** 3
    return y

forward_batch = vmap(forward, in_axes=(None,0))
theta = jnp.zeros(4,dtype=float)
print(forward_batch(theta,x_batch).shape)

def loss(theta,x_batch,y_label_batch):
    return jnp.mean(jnp.square(forward_batch(theta,x_batch) - y_label_batch))

grad = jax.value_and_grad(loss,argnums=0)
print(grad(theta, x_batch, y_label_batch)[1])

def step(lrate, theta, x_batch, y_label_batch):
    lossval, dldt = grad(theta, x_batch, y_label_batch)
    theta = jax.tree_util.tree_map(lambda t, dt: t-lrate*dt, theta, dldt)
    return theta, lossval

jit_step = jax.jit(step)

# initialise theta
theta = jnp.zeros(4,dtype=float)

# run gradient descent
for i in range(1000):
    theta, loss = jit_step(0.1,theta,x_batch,y_label_batch)
    print("{}: {:.4f}".format(i,loss))

x_output = jnp.linspace(-1, 1, 100).reshape((100, 1))
y_output = forward(theta,x_output)
plt.scatter(x_batch, y_label_batch, label="training data")
plt.plot(x_output, y_output, label="prediction")
plt.legend()
plt.xlabel("x");
plt.ylabel("y")
plt.show()
