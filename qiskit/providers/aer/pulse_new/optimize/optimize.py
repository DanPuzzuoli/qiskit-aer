from scipy.optimize import minimize
from jax import value_and_grad, jit, random, config
from random import randint
import jax.numpy as np
from jax.ops import index_update
import numpy as onp
from time import time
from joblib import Parallel, delayed
from copy import deepcopy
from qiskit.tools.parallel import CPU_COUNT

import os

# this was suggested in jax github issue to restrict XLA compilation to only one thread
# note the space after the first line needs to be there!
#os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
#                          "intra_op_parallelism_threads=1")
config.update("jax_enable_x64", True)

def normalize_objective_bfgs(obj,
                             input_shape,
                             output_len=None,
                             indices_to_normalize=None,
                             upper_limits=None,
                             lower_limits=None,
                             shots=1,
                             n_jobs=-1,
                             seed=None):
    """
    Given a vector valued function (assumes jax), normalizes the specified
    entries to lie between some bounds. This is done "empirically", i.e. the
    objective entries are individually minimized/maximized to find empirical
    bounds.

    Assumes the objective is a vector but may be useful to have more
    general indexing.

    Args:
        obj (function): objective function, takes in jax arrays of shape
                        input_shape, outputs 1d jax arrays of length output_len
        input_shape (tuple): shape of input array
        output_len (int): length of output
        indices_to_normalize (iterable): indices to normalize
        upper_limits (iterable): upper limits for objective entries
        lower_limits (iterable): lower  "
        shots (int): number of shots to use in normalization
        n_jobs (int): number of parallel processes
        seed (int): seed for random elements
    """

    if output_len is None:
        output_len = len(obj(np.zeros(input_shape)))

    if indices_to_normalize is None:
        indices_to_normalize = range(output_len)

    if upper_limits is None:
        upper_limits = np.ones(len(indices_to_normalize))

    if lower_limits is None:
        lower_limits = np.zeros(len(indices_to_normalize))

    start = time()

    print('Normalizing objective...')

    rng = onp.random.default_rng(seed)

    min_vals = []
    max_vals = []

    # find min and maximum of each component of the objective
    for idx in indices_to_normalize:
        print('Normalizing objective component: ' + str(idx))
        print('.... finding min')

        # find minimum
        min_obj = lambda x: obj(x)[idx]
        min_seed = rng.integers(0, np.iinfo(np.int32).max - 1)
        min_results = find_pulse_bfgs_parallel(min_obj,
                                               input_shape,
                                               shots=shots,
                                               n_jobs=n_jobs,
                                               seed=min_seed)
        min_vals.append(get_best_result(min_results).fun)
        print('.... min found: ' + str(min_vals[-1]))


        print('.... finding max')
        # find max
        max_obj = lambda x: -min_obj(x)
        max_seed = rng.integers(0, np.iinfo(np.int32).max - 1)
        max_results = find_pulse_bfgs_parallel(max_obj,
                                               input_shape,
                                               shots=shots,
                                               n_jobs=n_jobs,
                                               seed=max_seed)
        max_vals.append(-get_best_result(max_results).fun)
        print('.... max found: ' + str(max_vals[-1]))


    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)

    # transform limits of objective entries via linear function

    # get slope
    m = (upper_limits - lower_limits) / (max_vals - min_vals)

    # get offset
    b = lower_limits / m - min_vals

    def normalized_obj(x):
        output = obj(x)
        return index_update(output,
                            indices_to_normalize,
                            m * (output[indices_to_normalize] + b))

    print('\nTime taken: ' + str(time() - start))

    return normalized_obj


def get_best_result(results):
    """
    Return best optimization result from a list of results
    """
    func_vals = [r.fun for r in results]
    best_idx = func_vals.index(min(func_vals))
    return results[best_idx]



def find_pulse_bfgs_parallel(obj, ctrl_shape, shots=1, n_jobs=-1, seed=None):

    start = time()

    print('Total jobs: {}'.format(str(shots)))
    print('Jobs completed: 0', end="")

    # force initial compilation. This unfortunately doesn't work and
    # causes it to hang
    #mod_obj = jit(vec_value_and_grad(obj, ctrl_shape))
    #mod_obj(np.ones(ctrl_shape))

    # define optimization function
    def run_opt(shot_seed):
        rng = onp.random.default_rng(shot_seed)
        initial_guess = rng.uniform(size=ctrl_shape, low=-1, high=1)
        mod_obj = jit(vec_value_and_grad(obj, ctrl_shape))
        result = minimize(mod_obj,
                          initial_guess.flatten(),
                          method='BFGS',
                          jac=True,
                          options={'disp': False})

        result.x = result.x.reshape(ctrl_shape)
        return result

    rng = onp.random.default_rng(seed)
    shot_seeds = rng.integers(0, np.iinfo(np.int32).max - 1, shots)

    #
    if shots == 1:
        n_jobs == 1
    elif shots >= CPU_COUNT and n_jobs == -1:
        n_jobs = CPU_COUNT
    elif shots < CPU_COUNT and n_jobs == -1:
        n_jobs = shots

    results = Parallel(n_jobs=n_jobs)(delayed(run_opt)(shot_seed) for shot_seed in shot_seeds)
    end = time()

    time_taken = end - start

    print('\nTime taken: ' + str(end - start))
    return sorted(results, key=lambda x: x.fun)

def find_pulse_bfgs(obj, ctrl_shape, initial_guess=None, update_rate=None):
    """ Runs bfgs algorithm from scipy minimize for a given objective, ctrl_shape
    and initial guess
    """

    mod_obj = jit(vec_value_and_grad(obj, ctrl_shape))

    if update_rate is not None:
        mod_obj = updating_function(mod_obj, update_rate)

    if initial_guess == None:
        rng = onp.random.default_rng()
        initial_guess = rng.uniform(size=ctrl_shape, low=-1, high=1)

    # set the start time
    start = time()

    print('Optimizing pulse...')
    # run the optimization
    result = minimize(mod_obj, initial_guess.flatten(), method='BFGS', jac=True, options={'disp': True})
    # reshape the point the optimizer ends on to be the correct shape
    result.x = result.x.reshape(ctrl_shape)

    # record the end time and report the total time taken
    end = time()
    print('Total time taken: ' + str(end-start))

    return result

def vec_value_and_grad(f, ctrl_shape):
    """ Given a function f returns a version of value_and_grad(f) that takes 1d arrays
    and returns 1d array gradients (for scipy minimize)
    """

    f_value_and_grad = value_and_grad(f)

    def vecf(x):
        val, grad = f_value_and_grad(x.reshape(ctrl_shape))
        return val, grad.real.flatten()

    return vecf

def updating_function(f, update_rate):
    """Print updates at each function call
    """

    # set up the call counter
    calls = 0
    time_taken = 0.

    # define the new function
    def upd_f(x):
        # give the function access to the call counter, and increment it
        # every call
        nonlocal calls
        nonlocal time_taken
        calls = calls + 1

        start = time()
        #compute f
        output = f(x)
        end = time()

        time_taken += end - start

        # if its time to give an update, report the update depending on the
        # format of the output of f
        if calls % update_rate == 0:
            val = None
            if type(output) == tuple:
                val = output[0]
            else:
                val = output

            print('Evaluation {}: obj = {}, time = {}'.format(str(calls), str(val), str(time_taken)))
            time_taken = 0.
        return output

    return upd_f


# try this out
# taken from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
# code directly copied from post by Connor Clark
# Should probably be rewritten if ever used in qiskit
# I tried (naively) adding tracking of the best optimization value to this but
# some weird stuff started happening (e.g. reporting values of 1e-100 or
# the [trivial] logic of checking which value is actually smaller produced
# weird results)
from collections import defaultdict

# patch joblib progress callback
class BatchCompletionCallBack(object):
  completed = defaultdict(int)

  def __init__(self, time, index, parallel):
    self.index = index
    self.parallel = parallel

  def __call__(self, index):
    BatchCompletionCallBack.completed[self.parallel] += 1
    print("\rJobs completed: {}".format(BatchCompletionCallBack.completed[self.parallel]), end="")
    if self.parallel._original_iterator is not None:
      self.parallel.dispatch_next()

import joblib.parallel
joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack
