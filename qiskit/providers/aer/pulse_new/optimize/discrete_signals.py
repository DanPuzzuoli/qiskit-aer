"""Some functions for constructing/manipulating discrete signals."""

import jax.numpy as jnp
from jax import vmap
from jax.lax import conv_general_dilated
import numpy as np
from numpy.polynomial.legendre import Legendre
from qiskit.providers.aer.pulse_new.models.signals import PiecewiseConstant
from qiskit.providers.aer.pulse_new.models.transfer_functions import BaseTransferFunction

"""
General functions for creating constrained PiecewiseConstant signals from a
linear combination of basis functions
"""

def bounded_PiecewiseConstant(dt, coeffs, basis_arrays,
                              lb=-1, ub=1,
                              carrier_freq=0.,
                                  n_pad_start=0,
                                  n_pad_end=0,
                                  val_pad_start=0.,
                                  val_pad_end=0.):

    samples = bounded_linear_combo(coeffs, basis_arrays, lb, ub)
    samples = pad_array(samples, n_start=n_pad_start, pad_start=val_pad_start,
                                 n_end=n_pad_end, pad_end=val_pad_end)
    signal = PiecewiseConstant(dt=dt, samples=samples, carrier_freq=carrier_freq)

    return signal

def bounded_linear_combo(coeffs, basis_arrays, lb=-1, ub=1):
    """Returns a linear combo of basis_arrays, smoothly mapped into the
    requested bounds

    Args:
        coeffs: coefficients
        basis_arrays: list of basis arrays (as an array)
        lb: lower bound on output array values
        ub: upper bounded on output array values
    """
    linear_combo = jnp.tensordot(coeffs, basis_arrays, axes=1)
    return diffeo(linear_combo, [lb, ub])

"""
specific bases
"""

def discretized_legendre(degree, N, ran=[-1,1]):
    """
    Discretized Legendre polynomial of specified degree over interval [0,T], with samples taken
    using midpoint rule

    Args:
        degree (int): degree of Legendre polynomial (starting from 0)
        T (float): right limit of domain [0,T]
        N (int): number of steps in discretization
        ran (list): upper and lower bound on range of polynomial

    """
    dt = 1. / N
    coeffs = np.zeros(degree + 1, dtype=float)
    coeffs[-1] = 1.

    return jnp.array(Legendre(coeffs, [0,1.], ran)(np.linspace(0, 1. - dt, N) + (dt / 2)))

def discretized_legendre_basis(degrees, N, ran=[-1,1]):
    """Return all discretized legendre polynomials up to a given degree
    """

    if isinstance(degrees, int):
        degree_list = list(range(degrees + 1))
    elif isinstance(degrees, list):
        degree_list = degrees

    disc_leg_map = map(lambda deg: discretized_legendre(deg, N, ran), degree_list)

    # convert to array and return
    return jnp.array(list(disc_leg_map))

"""
convolutions
"""


class DiscreteEnvelopeFrequencyFilter(BaseTransferFunction):

    def __init__(self, sample_rate, freq_response_func):
        """
        Args:
            sample_rate: rate at which to resample input
            freq_response_func: vectorized callable for the coefficient
                                modification in fft
        """

        self._sample_rate = sample_rate
        self._freq_response_func = freq_response_func

    def apply(self, signal):

        dt = signal.dt / self._sample_rate
        times = jnp.arange(signal.duration * self._sample_rate) * dt + 0.5 * dt

        freqs = (1/dt)*jnp.fft.fftfreq(times.shape[-1])
        resampled_sig = expand_1d_array(signal._samples, self._sample_rate)
        sig_fft = jnp.fft.fft(resampled_sig)
        filtered_sig_fft = sig_fft * self._freq_response_func(freqs)
        filtered_sig = jnp.fft.ifft(filtered_sig_fft)

        return PiecewiseConstant(dt=dt,
                                 samples=filtered_sig,
                                 carrier_freq=signal.carrier_freq)

class DiscretePadSignal(BaseTransferFunction):

    def __init__(self, n_start=1, start_pad=0., n_end=1, end_pad=0.):
        self._n_start = n_start
        self._start_pad = start_pad
        self._n_end = n_end
        self._end_pad = end_pad

    def apply(self, signal):
        new_samples = pad_array(signal._samples,
                                self._n_start, self._start_pad,
                                self._n_end, self._end_pad)
        return PiecewiseConstant(dt=signal.dt, samples=new_samples, carrier_freq=signal.carrier_freq)

class DiscreteLowpassFilter(DiscreteEnvelopeFrequencyFilter):

    def __init__(self, sample_rate, cutoff):
        """
        Args:
            sample_rate: rate at which to resample input
            freq_response_func: vectorized callable for the coefficient
                                modification in fft
        """

        self._sample_rate = sample_rate
        self._freq_response_func = lambda f: (1/(1+1.0j*f/cutoff))

class DiscreteEnvelopeConvolution(BaseTransferFunction):

    def __init__(self, sample_rate, kernel_samples, pad_start=0., pad_end=0.):
        """
        Args:
            sample_rate: rate at which the kernel is sampled relative to
                         the input
            kernel_samples: kernel samples
            pad_start: value to pad at the start when performing convolution
            pad_end: value to pad at the end when performing convolution
        """

        self._sample_rate = sample_rate
        self._kernel_samples = kernel_samples
        self._pad_start = pad_start
        self._pad_end = pad_end

    def apply(self, signal):

        new_dt = signal.dt / self._sample_rate
        new_samples = fine_1d_conv(signal._samples,
                                   self._kernel_samples,
                                   self._sample_rate)

        return PiecewiseConstant(dt=new_dt,
                                 samples=new_samples,
                                 carrier_freq=signal.carrier_freq)




"""
array convolution functions
"""

def fine_1d_conv(in_x, kernel, kern_sample_rate=1, pad_start=0, pad_end=0):
    """Convolve input with a kernel, with kern_sample_rate giving the number of samples in
    kernel per sample of in_x.

    This function "expands" in_x by repeating each entry in in_x kern_sample_rate times,
    then convolves the expanded signal with kernel, padding with len(kernel) zeros. The purpose
    of the padding is to ensure that the convolved signal starts and ends with 0.

    Args:
        in_x (1d array): input signal
        kernel (1d array): kernel
        kern_sample_rate (int): number

    Returns:
        array: convolved signal
    """

    # expand the input
    # this may actually be doable with conv_general_dilated
    expanded = expand_1d_array(in_x, kern_sample_rate)
    kernel_len = len(kernel)

    # bad on either side
    expanded = pad_array(expanded, kernel_len, pad_start, kernel_len, pad_end)

    return conv_general_dilated(jnp.array([[expanded]]),
                                jnp.array([[kernel]]),
                                (1,),
                                padding=[(0,0)])[0,0]

def pad_array(in_x, n_start=1, pad_start=0., n_end=1, pad_end=0.):
    pre_padded = jnp.append(pad_start * jnp.ones(n_start), in_x)
    return jnp.append(pre_padded, pad_end * jnp.ones(n_end))

def expand_1d_array(in_x, k):
    """Expand an input array but repeating each entry k times.
    """
    return vmap(lambda x: x * jnp.ones(k))(in_x).flatten()


"""
standard diffeomorphism from R into some range
"""

def diffeo(x, image=[-1, 1]):
    width = image[1] - image[0]

    return jnp.arctan(x)/(jnp.pi/2) * (width / 2) + image[0] + (width / 2)
