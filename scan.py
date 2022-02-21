from functools import partial
import jax
import numpy
import time
import timeit

N = 10000

# Sum
def logp_sum(epsilon):
    return jax.numpy.sum(epsilon)

# Accumulation, x[n] = x[n - 1] + epsilon[n]
def logp_accum(epsilon):
    def lag1(carry_x, eps):
        x = carry_x + eps
        return x, x

    carry, y = jax.lax.scan(lag1, 0.0, epsilon)
    return jax.numpy.sum(y)

# Trickier accumulation, x[n] = 2 * x[n - 1] - x[n - 2] + epsilon[n]
def logp_trick_accum(epsilon):
    def lag3(carry_x, eps):
        x = 2 * carry_x[0] - carry_x[1] + eps
        return (x, carry_x[0]), x

    carry, y = jax.lax.scan(lag3, (0.0, 0.0), epsilon)
    return jax.numpy.sum(y)

rng = numpy.random.default_rng()

def time_function(func, epsilon):
    start = time.time()
    grad_vmap = jax.jit(jax.grad(lambda x : jax.numpy.sum(jax.vmap(func)(x))))
    vmap_grad_val = grad_vmap(epsilon)
    print(f"Compile time vector: {time.time() - start} s")

    start = time.time()
    def loop_f(epsilon):
        total = 0.0
        for i in range(epsilon.shape[0]):
            total += func(epsilon[i])
        return total
    grad_loop = jax.jit(jax.grad(loop_f))
    loop_grad_val = grad_loop(epsilon)
    print(f"Compile time loop: {time.time() - start} s")

    print(f"max diff in vmap and loop gradients: {numpy.abs(vmap_grad_val - loop_grad_val).max()}")

    timer = timeit.Timer(lambda : grad_vmap(epsilon))
    number, total_time = timer.autorange()
    time_per_iteration_vec = timer.timeit(number = number) / number

    timer = timeit.Timer(lambda : grad_loop(epsilon))
    number, total_time = timer.autorange()
    time_per_iteration_loop = timer.timeit(number = number) / number

    return time_per_iteration_vec * 1e6, time_per_iteration_loop * 1e6

for K in [1, 4, 64]:
    epsilon = rng.normal(size = K * N).reshape(K, N)
    starts = numpy.arange(K) * N
    stops = (numpy.arange(K) + 1) * N
    print(f"\nTimings: K = {K}:")
    print(f"{time_function(logp_sum, epsilon)} us (vec, loop) time per sum call")
    print(f"{time_function(logp_accum, epsilon)} us (vec, loop) time per accumulate call")
    print(f"{time_function(logp_trick_accum, epsilon)} us (vec, loop) time per tricky accumulate call")

def logp_trick_accum_reference(epsilon):
    N = epsilon.shape[0]
    output = numpy.zeros(N)
    for i in range(N):
        if i == 0:
            output[i] = epsilon[i]
        elif i == 1:
            output[i] = 2 * output[i - 1] + epsilon[i]
        else:
            output[i] = 2 * output[i - 1] - output[i - 2] + epsilon[i]
    return sum(output)
        
epsilon = rng.normal(size = N)
#print(logp_trick_accum([0], [N], jax.numpy.array(epsilon)))
#print(logp_trick_accum_reference(epsilon))