"""Some functions for constructing/manipulating discrete signals."""

import jax.numpy as np
from jax import vmap
from jax.lax import conv_general_dilated
from jax.ops import index_update, index
from numpy.polynomial.legendre import Legendre
from matplotlib import pyplot as plt

def diffeo(x, image=[-1, 1]):
    width = image[1] - image[0]

    return np.arctan(x)/(np.pi/2) * (width / 2) + image[0] + (width / 2)
