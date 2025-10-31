import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
import jax.numpy as jnp
from jax import grad,vmap,jit
import jax
import matplotlib.pyplot as plt

def fn(x):
    return jnp.tanh(x)

x = jnp.linspace(-5,5,500)

df = grad(fn)
d2f = grad(df)

plt.plot(x,fn(x),label="f(x)")
plt.plot(x,[df(x_) for x_ in x],label="df(x)")
plt.plot(x,[d2f(x_) for x_ in x],label="d2f(x)")
plt.legend()
plt.xlabel("x")
plt.show()

print(jax.make_jaxpr(fn)(x))# JAX transforms programs using a simple intermediate language call jaxpr
print(jax.make_jaxpr(vmap(df,in_axes=0))(x))

# Jacobian
jacobian_fn = jax.jacfwd(fn)
j = jacobian_fn(x)
print(j)
print(j.shape)

# vector-Jacobian product
f, vjp_fn = jax.vjp(fn, x)
dfdx, = vjp_fn(jnp.ones_like(x))

plt.plot(x, f, label="f(x)")
plt.plot(x, dfdx, label="df/dx")
plt.legend()
plt.xlabel("x")
plt.show()


#VECTORISATION
def forward_fn(w, b, x):
    x = w @ x + b
    x = jnp.tanh(x)
    return x

key = jax.random.key(seed=0)
key1, key2, key3 = jax.random.split(key, 3)
x = jax.random.normal(key1, (3,))
w = jax.random.normal(key2, (10,3))
b = jax.random.normal(key3, (10,))
y = forward_fn(w, b, x)
print(x.shape)
print(y.shape)

forward_batch_fn = vmap(forward_fn, in_axes=(None,None,0))

x_batch = jax.random.normal(key, (1000, 3))
y_batch = forward_batch_fn(w, b, x_batch)
print(x_batch.shape)
print(y_batch.shape)


#JIT COMPILING
def fn(x):
    return x + x*x + x*x*x

jit_fn = jit(fn)

x = jax.random.normal(key, (1000,1000))



