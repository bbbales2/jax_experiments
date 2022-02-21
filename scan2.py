from functools import partial
import jax
import numpy
import time
import timeit

N = 100

def accum(p1):
    indices = jax.numpy.arange(p1.shape[0] + 1)
    
    def scan_func(carry, n):
        x1, x2 = carry
        x = jax.numpy.where(n > 0, p1[n - 1], 0.0)
        return (x, x1), x

    carry, x = jax.lax.scan(scan_func, (0.0, 0.0), indices)

    return jax.numpy.sum(x)

rng = numpy.random.default_rng()

grad = jax.jit(jax.grad(accum))

print(grad(rng.normal(size = 5)))
