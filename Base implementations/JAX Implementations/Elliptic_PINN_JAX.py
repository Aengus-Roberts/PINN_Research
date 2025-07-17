import jax
import jax.numpy as jnp
from jax import grad, vmap
import optax
import matplotlib.pyplot as plt

# Define MLP model
def init_mlp(sizes, key):
    params = []
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        key, subkey = jax.random.split(key)
        W = jax.random.normal(subkey, (in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
        b = jnp.zeros(out_dim)
        params.append((W, b))
    return params

def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, W) + b)
    W, b = params[-1]
    return jnp.dot(x, W) + b

# Define residual of the PDE: -epsilon^2 u'' + u = 1
def residual(params, x, epsilon):
    u = lambda x_: mlp(params, x_)
    du = grad(u)
    d2u = grad(du)
    return -epsilon**2 * d2u(x) + u(x) - 1.0

# Full loss including residual and boundary conditions
def loss_fn(params, x_colloc, epsilon):
    res = vmap(lambda x: residual(params, x, epsilon))(x_colloc)
    loss_r = jnp.mean(res**2)

    u = lambda x: mlp(params, x)
    bc1 = u(jnp.array([[-1.0]]))
    bc2 = u(jnp.array([[1.0]]))
    loss_bc = bc1**2 + bc2**2

    return loss_r + loss_bc

def train(params, x_colloc, epsilon, steps=5000, lr=1e-3):
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_colloc, epsilon)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(steps):
        params, opt_state, loss = step(params, opt_state)
        if i % 500 == 0:
            print(f"Step {i}: Loss = {loss:.5e}")
    return params

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    net_sizes = [1, 64, 64, 1]
    params = init_mlp(net_sizes, key)

    x_colloc = jnp.linspace(-1, 1, 100).reshape(-1, 1)
    epsilon = 0.01

    params = train(params, x_colloc, epsilon)

    x_test = jnp.linspace(-1, 1, 200).reshape(-1, 1)
    y_pred = vmap(lambda x: mlp(params, x))(x_test)

    plt.plot(x_test, y_pred, label="PINN Prediction")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True)
    plt.legend()
    plt.title("PINN Solution to -eps^2 u'' + u = 1")
    plt.show()