from jax import jit, vmap, grad, value_and_grad, random, numpy as jnp
import optax
import matplotlib.pyplot as plt

#GLOBAL DEFINES
seed = 0
EPSILON = 0.01
opt = optax.lbfgs()

#Initialising Multi-Layer Perceptron (MLP)
def init_mlp(sizes, key):
    params = []
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        key, subkey = random.split(key)
        W = random.normal(subkey, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)  # He initialization
        #W = jnp.ones(shape=(in_dim, out_dim))
        b = jnp.zeros(out_dim)
        params.append((W, b))
    return params

#Defining Forward Pass through MLP
def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, W) + b)  # Apply tanh to hidden layers
    W, b = params[-1]
    return jnp.dot(x, W) + b  # Output layer is linear

#Defining Loss
def residual(params, x):
    x = jnp.squeeze(x)  # Ensure scalar input
    u = lambda x_: mlp(params, x_.reshape(1, 1)).squeeze()
    du = grad(u)
    d2u = grad(du)
    return -EPSILON**2 * d2u(x) + u(x) - 1.0

def loss_fn(params, x_colloc):
    res = vmap(lambda x: residual(params, x))(x_colloc)
    loss_r = jnp.mean(res**2)

    # Dirichlet BCs at 0 and 1
    u = lambda x: mlp(params, x)[0]
    bc1 = u(jnp.array([0.0]))
    bc2 = u(jnp.array([1.0]))
    loss_bc = bc1**2 + bc2**2

    return loss_r + loss_bc

# Training step
@jit
def train_step(params, opt_state, x):
    loss, grads = value_and_grad(loss_fn)(params, x)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def main():
    key = random.PRNGKey(seed)
    sizes = [1, 64, 64, 1]
    params = init_mlp(sizes, key)

    # Uniform collocation points in [0, 1]
    x_colloc = jnp.linspace(0.0, 1.0, 100).reshape(-1, 1)

    opt_state = opt.init(params)

    for i in range(5000):
        params, opt_state, loss = train_step(params, opt_state, x_colloc)
        if i % 500 == 0:
            print(f"Step {i}, Loss: {loss:.6f}")

    # Prediction and plotting
    x_test = jnp.linspace(0.0, 1.0, 200).reshape(-1, 1)
    u_pred = vmap(lambda x: mlp(params, x))(x_test)

    plt.plot(x_test, u_pred, label="PINN Prediction")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True)
    plt.title("PINN Solution to -eps^2 u'' + u = 1")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()