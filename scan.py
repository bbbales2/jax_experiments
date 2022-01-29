from functools import partial
import jax
import numpy
import time
import timeit

N = 10000

# Sum
def logp_sum(starts, stops, epsilon):
    total = 0.0
    for start, stop in zip(starts, stops):
        total += jax.numpy.sum(epsilon[start:stop])
    return total

# Accumulation, x[n] = x[n - 1] + epsilon[n]
def logp_accum(starts, stops, epsilon):
    def lag1(carry_x, eps):
        x = carry_x + eps
        return x, x

    total = 0.0
    for start, stop in zip(starts, stops):
        carry, y = jax.lax.scan(lag1, 0.0, epsilon[start:stop])
        total += jax.numpy.sum(y)
    return total

# Trickier accumulation, x[n] = 2 * x[n - 1] - x[n - 2] + epsilon[n]
def logp_trick_accum(starts, stops, epsilon):
    def lag3(carry_x, eps):
        x = 2 * carry_x[0] - carry_x[1] + eps
        return (x, carry_x[0]), x

    total = 0.0
    for start, stop in zip(starts, stops):
        carry, y = jax.lax.scan(lag3, (0.0, 0.0), epsilon[start:stop])
        total += jax.numpy.sum(y)
    return total

rng = numpy.random.default_rng()

def time_function(func, epsilon):
    start = time.time()
    grad = jax.jit(jax.grad(func))
    grad(epsilon)
    print(f"Compile time: {time.time() - start} s")

    timer = timeit.Timer(lambda : grad(epsilon))
    number, total_time = timer.autorange()
    time_per_iteration = timer.timeit(number = number) / number
    return time_per_iteration * 1e6

epsilon = rng.normal(size = N)
starts = [0]
stops = [N]
print("Scalar timings:")
print(f"{time_function(partial(logp_sum, starts, stops), epsilon)} us time per sum call")
print(f"{time_function(partial(logp_accum, starts, stops), epsilon)} us time per accumulate call")
print(f"{time_function(partial(logp_trick_accum, starts, stops), epsilon)} us time per tricky accumulate call")

M = 5
epsilon = rng.normal(size = N)
starts = [0] * M
stops = [N] * M
print(f"\nSmall vector ({M}) timings:")
print(f"{time_function(partial(logp_sum, starts, stops), epsilon)} us time per sum call")
print(f"{time_function(partial(logp_accum, starts, stops), epsilon)} us time per accumulate call")
print(f"{time_function(partial(logp_trick_accum, starts, stops), epsilon)} us time per tricky accumulate call")

M = 100
epsilon = rng.normal(size = N)
starts = [0] * M
stops = [N] * M
print(f"\nLarge vector ({M}) timings:")
print(f"{time_function(partial(logp_sum, starts, stops), epsilon)} us time per sum call")
print(f"{time_function(partial(logp_accum, starts, stops), epsilon)} us time per accumulate call")
print(f"{time_function(partial(logp_trick_accum, starts, stops), epsilon)} us time per tricky accumulate call")

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