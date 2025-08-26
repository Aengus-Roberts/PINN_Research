"""
ENGD Optimization.
Two dimensional Elliptic Boundary Layer equation example. Solution given by

u(x,y) = (Ae^x/ε + Be^-x/ε + 1)(Ae^y/ε + Be^-y/ε + 1)

"""
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit

from ngrad.models import init_params, mlp
from ngrad.domains import Interval, IntervalBoundary
from ngrad.integrators import DeterministicIntegrator
from ngrad.utility import laplace, grid_line_search_factory
from ngrad.inner import model_laplace, model_identity
from ngrad.gram import gram_factory, nat_grad_factory

jax.config.update("jax_enable_x64", True)

# random seed
seed = 0

# Problem Constants
epsilon = 0.01
A = (jnp.exp(-1/epsilon)-1)/(2*jnp.sinh(1/epsilon))
B = - 1 - A

# domains
interior = Interval(0,1)
boundary = IntervalBoundary(0,1)

# integrators
interior_integrator = DeterministicIntegrator(interior, 30)
boundary_integrator = DeterministicIntegrator(boundary, 2)
eval_integrator = DeterministicIntegrator(interior, 200)

# model
activation = lambda x: jnp.tanh(x)
layer_sizes = [1, 32, 1]
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


# gramians
gram_bdry = gram_factory(
    model=model,
    trafo=model_identity,
    integrator=boundary_integrator
)

gram_laplace = gram_factory(
    model=model,
    trafo=model_laplace,
    integrator=interior_integrator
)


@jit
def gram(params):
    return gram_laplace(params) + gram_bdry(params)


# natural gradient
nat_grad = nat_grad_factory(gram)

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


# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5 ** grid
ls_update = grid_line_search_factory(loss, steps)

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))
v_error_abs_grad = vmap(
    lambda x: jnp.dot(grad(error)(x), grad(error)(x)) ** 0.5
)


def l2_norm(f, integrator):
    return integrator(lambda x: (f(x)) ** 2) ** 0.5


# natural gradient descent with line search
for iteration in range(51):
    grads = grad(loss)(params)
    nat_grads = nat_grad(params, grads)
    params, actual_step = ls_update(params, nat_grads)

    if iteration % 5 == 0:
        # errors
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)

        print(
            f'NG Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error} and error H1: {h1_error} and step: {actual_step}'
        )