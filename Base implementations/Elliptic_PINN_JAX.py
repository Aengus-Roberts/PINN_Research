import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from jax import grad, vmap, jit

EPSILON = .01

# Initialize a simple 2-hidden-layer neural network
def init_params(key, layers=[1, 20, 1]):
    params = []
    keys = jax.random.split(key, len(layers))
    for m, n, k in zip(layers[:-1], layers[1:], keys):
        W = jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m)
        b = jnp.zeros((n,))
        params.append((W, b))
    return params

# Forward pass through the network
def forward(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, W) + b)
    W, b = params[-1]
    return jnp.dot(x, W) + b

# Compute the PINN loss: PDE residual + boundary loss
def loss_fn(params, x):
    def u(x):
        return forward(params, x.reshape(1, -1))[0, 0]

    dudx = grad(u)
    d2udx2 = grad(dudx)
    u_xx = vmap(d2udx2)(x)
    u_pred = vmap(u)(x)

    residual = -(EPSILON ** 2) * u_xx + u_pred - 1
    residual_loss = jnp.mean(residual ** 2)

    # Boundary loss
    bc_loss = u(jnp.array([0.0]))**2 + u(jnp.array([1.0]))**2

    return residual_loss + bc_loss

# Training loop
def train():
    key = jax.random.PRNGKey(0)
    params = init_params(key)

    x_train = jnp.linspace(0, 1, 100).reshape(-1, 1)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_train)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for epoch in range(5000):
        params, opt_state, loss = step(params, opt_state)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6e}")

    return params

# Plot the result
def plot_solution(params):
    x_test = jnp.linspace(0, 1, 200).reshape(-1, 1)
    u_pred = vmap(lambda x: forward(params, x.reshape(1, -1))[0, 0])(x_test)
    #u_true = jnp.sin(jnp.pi * x_test).flatten()

    plt.plot(x_test, u_pred, label="PINN", linestyle='--')
    #plt.plot(x_test, u_true, label="Exact", linestyle='-')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    #plt.title("PINN solution of -u''(x) = π² sin(πx)")
    plt.show()

if __name__ == "__main__":
    trained_params = train()
    plot_solution(trained_params)