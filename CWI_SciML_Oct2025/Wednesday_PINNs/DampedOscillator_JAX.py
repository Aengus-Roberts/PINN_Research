import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import optax
import time
from functools import partial

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def exact_solution(d, w0, t):
    "Computes the analytical solution to the under-damped harmonic oscillator."
    assert d < w0
    w = jnp.sqrt(w0 ** 2 - d ** 2)
    phi = jnp.arctan(-d / w)
    A = 1 / (2 * jnp.cos(phi))
    cos = jnp.cos(phi + w * t)
    exp = jnp.exp(-d * t)
    u = exp * 2 * A * cos
    return u


def plot_result(i, t_physics_batch, t_boundary, t_test_batch, u_exact_batch, u_test_batch):
    "Plots the PINN's prediction"
    test_error = jnp.mean(jnp.abs(u_test_batch - u_exact_batch) / u_exact_batch.std())
    plt.figure(figsize=(8, 3))
    plt.scatter(t_physics_batch[:, 0], jnp.zeros_like(t_physics_batch)[:, 0], s=20, lw=0, color="tab:blue", alpha=0.6,
                label="Collocation points")
    plt.scatter(t_boundary, 0, s=20, lw=0, color="tab:red", alpha=0.6, label="Boundary point")
    plt.plot(t_test_batch[:, 0], u_exact_batch[:, 0], label="Exact solution", color="tab:grey", alpha=0.6)
    plt.plot(t_test_batch[:, 0], u_test_batch[:, 0], label="PINN solution", color="tab:green")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"Training step {i + 1}    Relative L1 test error: {test_error:.2%}")
    plt.legend()
    plt.show()


class FCN:
    """Fully connected network in JAX, using only jax.numpy

    Note for the curious reader: any functions which are designed for JAX transformations (vmap, grad, jit, etc)
    should not include side-effects (a side-effect is any effect of a function that doesn’t appear in its output).

    But standard python class methods (e.g. def __init__(self, ..)) often adjust `self` outside of the method.
    This is risky, unless you know exactly what you are doing; and is why we only define static methods here.

    This means the FCN class is really just a collection of functions, i.e. a convienent namespace,
    and we carry around the state of the network (i.e. its `parameters`, normally contained in `self` in PyTorch)
    explicitly in our JAX code, passing it explicitly to each method. See here for more discussion:
    https://docs.jax.dev/en/latest/stateful-computations.html
    """

    @staticmethod
    def init_parameters(key, layer_sizes):
        """Initialise the parameters of the network.
        Parameters:
            key: current JAX RNG state
            layer_sizes: list defining the number of layers and the number of channels per layer, including
            input/output layers, e.g. [1,16,16,1]
        Returns:
            parameters: list of randomly initialised weights and biases [(W0, b0), ...]

        Note: JAX uses explicit random seed management; so we need to pass the current RNG state (`key`) explicitly
        to any `jax.random` calls, and split the RNG state by hand.
        """

        keys = jax.random.split(key, len(layer_sizes) - 1)  # split the key
        parameters = [FCN._random_layer_parameters(k, m, n)
                      for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]  # initialise all layers

        return parameters

    @staticmethod
    def _random_layer_parameters(key, m, n):
        "Randomly intialise the parameters of a single layer"

        W_key, b_key = jax.random.split(key)
        v = jnp.sqrt(1 / m)  # LeCun initialisation
        W = jax.random.uniform(W_key, (n, m), minval=-v, maxval=v)
        b = jax.random.uniform(b_key, (n,), minval=-v, maxval=v)

        return W, b

    @staticmethod
    def forward(parameters, x):
        """Forward pass of the network.
        Parameters:
            parameters: list of weights and biases [(W0,b0), ...]
            x: SINGLE input point of shape (xdim=layer_sizes[0],)
        Returns:
            u: SINGLE output point of shape (udim=layer_sizes[-1],)

        Note: a key philosophical difference between JAX and PyTorch is that in JAX we define non-batched
        operations (like the forward pass of the network with a SINGLE input point), and then use `vmap`
        afterwards to define the batched version (e.g. for evaluating the network over many points).
        With experience, this is a powerful way to build sophisticated, peformant, flexible code which remains
        readable and is less bug-prone.
        """

        assert x.ndim == 1
        activation_fn = jnp.tanh

        for W, b in parameters[:-1]:  # W has shape (m, n), b has shape (n,)
            x = jnp.dot(W, x) + b
            x = activation_fn(x)
        W, b = parameters[-1]
        u = jnp.dot(W, x) + b

        assert u.ndim == 1
        return u


# set random seed
key = jax.random.PRNGKey(0)

# define the neural network architecture and initialise its parameters
network = FCN
parameters = network.init_parameters(key, [1, 32, 32, 1])


t_test_batch = jnp.linspace(0, 1, 300).reshape(-1, 1)
u_test_batch = jax.vmap(network.forward, in_axes=(None, 0))(parameters, t_test_batch)
plt.figure(figsize=(8, 3))
plt.plot(t_test_batch[:, 0], u_test_batch[:, 0], label="Untrained network output", color="tab:green")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


def PINN_physics_loss(parameters, t, network, mu, k):
    """Compute the physics loss for the 1D harmonic oscillator problem.
    Parameters:
        parameters: network parameters
        t: SINGLE input point of shape (1,)
        network: network class
        mu: coefficient of friction
        k: spring constant
    Returns:
        loss_physics: SINGLE SCALAR loss value of shape ()

    ODE:
    d^2 u      du
    ----- + mu -- + ku = 0
    dt^2       dt

    Boundary conditions:
    u (0) = 1
    u'(0) = 0
    """

    assert t.shape == (1,)

    def u_fn(t):
        """Calls network.forward, isolating u as a scalar function of a scalar input
        suitable for jax.grad"""
        return network.forward(parameters, t.reshape(1, )).squeeze()  # has shape ()

    t = t.squeeze()  # has shape ()

    u = u_fn(t)  # has shape ()


    dudt_fn = jax.grad(u_fn)
    dudt = dudt_fn(t)

    d2udt2_fn = jax.grad(dudt_fn)
    d2udt2 = d2udt2_fn(t)

    loss_physics = (d2udt2 + mu * dudt + k * u) ** 2


    assert loss_physics.shape == ()
    return loss_physics


def PINN_boundary_losses(parameters, t, network):
    """Compute the two boundary losses for the 1D harmonic oscillator problem.
    Parameters:
        parameters: network parameters
        t: SINGLE input point of shape (1,)
        network: network class
    Returns:
        loss_displacement: SINGLE SCALAR loss value of shape ()
        loss_velocity: SINGLE SCALAR loss value of shape ()

    ODE:
    d^2 u      du
    ----- + mu -- + ku = 0
    dt^2       dt

    Boundary conditions:
    u (0) = 1
    u'(0) = 0
    """

    assert t.shape == (1,)

    def u_fn(t):
        """Calls network.forward, isolating u as a scalar function of a scalar input
        suitable for jax.grad"""
        return network.forward(parameters, t.reshape(1, )).squeeze()  # has shape ()

    t = t.squeeze()  # has shape ()

    u = u_fn(t)
    dudt_fn = jax.grad(u_fn)
    dudt = dudt_fn(t)
    loss_displacement, loss_velocity = (u - 1) ** 2, (dudt - 0) ** 2

    assert loss_displacement.shape == loss_velocity.shape == ()
    return loss_displacement, loss_velocity


def PINN_loss_batch(parameters, t_boundary, t_physics_batch, network, mu, k):
    """Computes the total PINN loss for the harmonic oscillator problem, across a batch of collocation points.
    Parameters:
        parameters: network parameters
        t_boundary: SINGLE input point of shape (1,)
        t_physics_batch: BATCH of collocation points of shape (N, 1)
        network: network class
        mu: coefficient of friction
        k: spring constant
    Returns:
        loss: SINGLE SCALAR loss value of shape ()
    """

    assert t_boundary.shape == (1,) and t_physics_batch.ndim == 2

    loss_physics = jnp.mean(jax.vmap(PINN_physics_loss, in_axes=(None, 0, None, None, None))(
        parameters, t_physics_batch, network, mu, k))

    loss_displacement, loss_velocity = PINN_boundary_losses(parameters, t_boundary, network)

    assert loss_physics.shape == loss_displacement.shape == loss_velocity.shape == ()

    # sum all the losses together, weighting terms appropriately
    loss = 1e-4 * loss_physics + loss_displacement + 1e-1 * loss_velocity

    return loss


@partial(jax.jit, static_argnums=(1, 5))  # JAX best practice is to only `jit` your highest-level function
def PINN_step(opt_state, optimiser, parameters, t_boundary, t_physics_batch, network, mu, k):
    "Updates PINN parameters using the `optax` Adam optimiser"

    # get loss and gradient over batch
    loss, grads = jax.value_and_grad(PINN_loss_batch, argnums=0)(
        parameters, t_boundary, t_physics_batch, network, mu, k)

    # apply parameter update
    updates, opt_state = optimiser.update(grads, opt_state, parameters)
    parameters = optax.apply_updates(parameters, updates)

    return loss, opt_state, parameters

# define boundary point, for the boundary loss
t_boundary = jnp.array([0.])  # has shape (1,)

# define batch of training points over the entire problem domain [0,1], for the physics loss
t_physics_batch = jnp.linspace(0, 1, 50).reshape(-1, 1)  # has shape (50, 1)

# define ODE parameters
d, w0 = 2, 40
mu, k = 2 * d, w0 ** 2

# get dense test points and exact solution to compare to
t_test_batch = jnp.linspace(0, 1, 300).reshape(-1, 1)
u_exact_batch = jax.vmap(exact_solution, in_axes=(None, None, 0))(d, w0, t_test_batch)

# define the optimiser
optimiser = optax.adam(learning_rate=1e-2)
opt_state = optimiser.init(parameters)

# start training
start = time.time()
for i in range(15000):

    # update parameters
    loss, opt_state, parameters = PINN_step(
        opt_state, optimiser, parameters, t_boundary, t_physics_batch, network, mu, k)

    # plot the result as training progresses
    if (i + 1) % 5000 == 0 or i == 0:
        # get PINN prediction and plot
        u_test_batch = jax.vmap(network.forward, in_axes=(None, 0))(parameters, t_test_batch)
        plot_result(i, t_physics_batch, t_boundary, t_test_batch, u_exact_batch, u_test_batch)

print(f"Total training time: {time.time() - start} seconds")
