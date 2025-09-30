# pip install jax jaxlib optax matplotlib
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import optax
import matplotlib.pyplot as plt

EPSILON = 0.1

# -------------------------------
# Knot set (clustered near 0 and 1 as in your PyTorch code)
# -------------------------------
def get_knot_points(N=50, eps=EPSILON):
    start = np.linspace(0.0, eps, N, endpoint=False)
    mid   = np.linspace(eps, 1.0 - eps, N, endpoint=False)
    end   = np.linspace(1.0 - eps, 1.0, N)
    knots = np.concatenate([start, mid, end])
    return jnp.asarray(knots, dtype=jnp.float32)

# -------------------------------
# Quadrature points
# -------------------------------
def get_quad_points(N=50, kind="uniform"):
    if kind == "uniform":
        # match your 3*N uniform sampling
        M = 3 * N
        x = jnp.linspace(0.0, 1.0, M, dtype=jnp.float32)
        w = jnp.ones_like(x) / M
        return x, w
    elif kind == "gauss":
        # numpy.leggauss gives nodes/weights on [-1,1]
        nodes, weights = np.polynomial.legendre.leggauss(N)
        x = (nodes + 1.0) / 2.0
        w = weights / 2.0
        return jnp.asarray(x, jnp.float32), jnp.asarray(w, jnp.float32)
    else:
        raise ValueError("Unsupported quadrature method")

# -------------------------------
# Model: y(x) = sum_i alpha_i ReLU(x - k_i)
# -------------------------------
def init_params(key, knots):
    K = knots.shape[0]
    # match PyTorch: random normal init for alpha
    alpha = jax.random.normal(key, shape=(K,), dtype=jnp.float32)
    return {"alpha": alpha}

def y_scalar(params, knots, x):
    # x is scalar; returns scalar
    relu = jnp.maximum(x - knots, 0.0)        # (K,)
    return jnp.dot(params["alpha"], relu)      # scalar

# batched evaluation over x vector
def y_batch(params, knots, x_vec):
    # x_vec: (N,)
    relus = jnp.maximum(x_vec[:, None] - knots[None, :], 0.0)  # (N, K)
    return relus @ params["alpha"]                             # (N,)

# -------------------------------
# Energy functional (Deep Ritz)
# E = ∫ [ (ε^2/2) (y')^2 + (1/2) y^2 - y ] dx  + BC penalty
# -------------------------------
def energy_loss(params, knots, xq, wq, epsilon):
    # y' via autodiff of scalar y
    dy_scalar = grad(lambda t: y_scalar(params, knots, t))
    dy = vmap(dy_scalar)(xq)                # (N,)
    y  = y_batch(params, knots, xq)         # (N,)

    integrand = 0.5 * (epsilon**2) * dy**2 + 0.5 * y**2 - y
    quad = jnp.sum(wq * integrand)

    # Dirichlet BCs: u(0)=u(1)=0
    u0 = y_scalar(params, knots, jnp.array(0.0, jnp.float32))
    u1 = y_scalar(params, knots, jnp.array(1.0, jnp.float32))
    bc = u0**2 + u1**2

    return quad + bc

# jit-compiled training step
#@jit
def train_step(params, opt_state, knots, xq, wq, epsilon, optimizer):
    loss, grads = jax.value_and_grad(energy_loss)(params, knots, xq, wq, epsilon)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def train(knots, xq, wq, epsilon=EPSILON, steps=10000, lr=1e-2, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_params(key, knots)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    for it in range(steps):
        params, opt_state, loss = train_step(params, opt_state, knots, xq, wq, epsilon, optimizer)
        if it % 500 == 0:
            print(f"step {it:5d} | loss = {loss:.6e}")
    return params

# -------------------------------
# True solution for comparison
# -ε^2 u'' + u = 1, u(0)=u(1)=0  on [0,1]
# u(x) = 1 - cosh((x-1/2)/ε) / cosh(1/(2ε))
# -------------------------------
def true_solution(x, eps=EPSILON):
    return 1.0 - np.cosh((x - 0.5) / eps) / np.cosh(1.0 / (2.0 * eps))

# -------------------------------
# Run and plot
# -------------------------------
def main():
    knots = get_knot_points(N=50, eps=EPSILON)

    # Test grid
    x_test = jnp.linspace(0.0, 1.0, 200, dtype=jnp.float32)

    # True solution
    y_true = jnp.asarray([true_solution(float(x), EPSILON) for x in x_test], dtype=jnp.float32)

    # Uniform quadrature
    x_uni, w_uni = get_quad_points(N=50, kind="uniform")
    params_uni = train(knots, x_uni, w_uni, epsilon=EPSILON, steps=10000, lr=1e-2, seed=0)
    y_uni = y_batch(params_uni, knots, x_test)

    # Gauss–Legendre quadrature
    x_g, w_g = get_quad_points(N=50, kind="gauss")
    params_g = train(knots, x_g, w_g, epsilon=EPSILON, steps=10000, lr=1e-2, seed=1)
    y_g = y_batch(params_g, knots, x_test)

    # Plot
    plt.figure(figsize=(7,4))
    plt.plot(x_test, y_true, label="True", color="green", linewidth=2.0)
    plt.plot(x_test, np.array(y_uni), "--", label="Uniform (Deep Ritz, ReLU knots)", color="red")
    plt.plot(x_test, np.array(y_g), "--", label="Gaussian (Deep Ritz, ReLU knots)", color="blue")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(rf"$-\varepsilon^2 u''(x) + u(x) = 1,\ \varepsilon = {EPSILON:.3f}$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()