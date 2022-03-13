from functools import partial
import jax
import numpy
import time
import timeit

N = 10000

def accum_scan(p1):
    indices = jax.numpy.arange(p1.shape[0])
    
    def scan_func(carry, n):
        x1, x2 = carry
        x = jax.scipy.stats.norm.logpdf(p1[n], 0.7)#jax.numpy.where(n > -1, , 0.0)
        #return jax.numpy.concatenate([jax.numpy.array([x]), carry[1:]]), x
        return (x, x1), x

    _, x = jax.lax.scan(scan_func, (0.0, 0.0), indices, unroll = 2)

    return jax.numpy.sum(x)

def accum_array(p1):
    indices = jax.numpy.arange(p1.shape[0])
    
    def scan_func(carry, n):
        x1, x2 = carry
        x = jax.scipy.stats.norm.logpdf(p1[n], 0.7)#jax.numpy.where(n > -1, , 0.0)
        return jax.numpy.concatenate([jax.numpy.array([x]), carry[1:]]), x

    _, x = jax.lax.scan(scan_func, jax.numpy.zeros(2), indices, unroll = 2)

    return jax.numpy.sum(x)

def accum(p1):
    return jax.numpy.sum(jax.scipy.stats.norm.logpdf(p1, 0.7))

rng = numpy.random.default_rng()

grad_array = jax.jit(jax.grad(accum_array))
grad_scan = jax.jit(jax.grad(accum_scan))
grad = jax.jit(jax.grad(accum))

test_x = rng.normal(size = N)

print(jax.jit(accum_array)(test_x))
print(jax.jit(accum_scan)(test_x))
print(jax.jit(accum)(test_x))

print(grad_array(test_x))
print(grad_scan(test_x))
print(grad(test_x))

print(timeit.timeit(lambda : grad_array(test_x), number = 1000))
print(timeit.timeit(lambda : grad_scan(test_x), number = 1000))
print(timeit.timeit(lambda : grad(test_x), number = 1000))
