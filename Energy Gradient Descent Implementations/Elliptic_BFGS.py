"""
BFGS Optimization.
Two dimensional Elliptic Boundary Layer equation example. Solution given by

u(x,y) = (Ae^x/ε + Be^-x/ε + 1)(Ae^y/ε + Be^-y/ε + 1)

"""

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import jaxopt
from jax.flatten_util import ravel_pytree

from ngrad.models import init_params, mlp
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
from ngrad.utility import laplace

jax.config.update("jax_enable_x64", True)

# random seed
seed = 0

# Problem Constants
epsilon = 0.01
A = (jnp.exp(-1/epsilon)-1)/(2*jnp.sinh(1/epsilon))
B = - 1 - A

# domains
interior = Square(1.)
boundary = SquareBoundary(1.)

# integrators
interior_integrator = DeterministicIntegrator(interior, 30)
boundary_integrator = DeterministicIntegrator(boundary, 30)
eval_integrator = DeterministicIntegrator(interior, 200)

# model
activation = lambda x: jnp.tanh(x)
layer_sizes = [2, 32, 1]
params = init_params(layer_sizes, random.PRNGKey(seed))
model = mlp(activation)
v_model = vmap(model, (None, 0))


# solution
@jit
def u_star(x):
    return jnp.prod(A*jnp.exp(x/epsilon) + B*jnp.exp(-x/epsilon) + 1)


# rhs
@jit
def f(x):
    return 1


# compute residual
epsilon_laplace_model = lambda params: laplace(lambda x: - epsilon**2 * model(params, x))
residual = lambda params, x: (epsilon_laplace_model(params)(x) + model(params,x) - f(x)) ** 2.
v_residual = jit(vmap(residual, (None, 0)))


# loss
@jit
def interior_loss(params):
    return interior_integrator(lambda x: v_residual(params, x))


@jit
def boundary_loss(params):
    return boundary_integrator(lambda x: v_model(params, x) ** 2)


@jit
def loss(params):
    return interior_loss(params) + boundary_loss(params)


# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))
v_error_abs_grad = vmap(
    lambda x: jnp.dot(grad(error)(x), grad(error)(x)) ** 0.5
)


def l2_norm(f, integrator):
    return integrator(lambda x: (f(x)) ** 2) ** 0.5


# optimizer settings
flat_params, unravel = ravel_pytree(params)
flat_loss = lambda x: loss(unravel(x))

BFGS = jaxopt.BFGS(
    fun=flat_loss,
    value_and_grad=False,
)
state = BFGS.init_state(flat_params)

# BFGS optimization
for iteration in range(200):
    grads = grad(loss)(params)
    flat_params, state = BFGS.update(flat_params, state)
    params = unravel(flat_params)

    if iteration % 1 == 0:
        # errors
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)

        print(
            f'BFGS Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error} and error H1: {h1_error}.'
        )