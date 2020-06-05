# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""DE methods."""

from abc import ABC, abstractmethod
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from type_utils import StateTypeConverter


class ODE_Method(ABC):
    """Abstract wrapper class for an ODE solving method.

    Class Attributes:
        - method_type_spec: dict specifying expected input type of the underlying method to
                            to facilitate automatic conversions (e.g. if the underlying method
                            requires 1d arrays, or only works with real arrays). Currently, only
                            conversion from 2d to 1d arrays is supported.

    Instance attributes:
        - _t, t: private and public time variable
        - _y, y: private and public state variable
        - rhs: dict with rhs-related functions.
    """

    # Class attribute specifying if the method requires the state of the DE to be a vector
    # In the future this can be generalized for more involved type handling (e.g. complex to real)
    method_spec = {'inner_state_spec': {'type': 'array'}}

    def __init__(self, t0=None, y0=None, rhs=None, solver_options={}):

        # set_options should be first as options may influence the behaviour of other functions
        self.set_options(solver_options)

        self._t = t0
        self.set_y(y0, reset=False)
        self.set_rhs(rhs)

    def integrate_over_interval(self, y0, interval, rhs=None):
        """Integrate over an interval, with additional options to reset the rhs functions.

        Args:
            y0 (array): state at the start of the interval
            interval (tuple or list): initial and start time, e.g. (t0, tf)
            rhs (callable or dict): Either the rhs function itself, or a dict of rhs-related
                                    functions

        Returns:
            state of the solver at the end of the integral
        """
        t0 = interval[0]
        tf = interval[1]

        self._t = t0
        self.set_y(y0, reset=False)
        if rhs is not None:
            self.set_rhs(rhs, reset=False)

        self._reset_method()

        self.integrate(tf)

        return self.y

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, new_t):
        self._t = new_t
        self._reset_method()

    @property
    def y(self):
        return self._state_type_converter.inner_to_outer(self._y)

    @y.setter
    def y(self, new_y):
        self.set_y(new_y)

    def set_y(self, new_y, reset=True):
        """Method for logic of setting internal state of solver with more control
        """
        type_spec = self.method_spec.get('inner_state_spec')
        self._state_type_converter = \
                        StateTypeConverter.from_outer_instance_inner_type_spec(new_y, type_spec)

        self._y = self._state_type_converter.outer_to_inner(new_y)

        self._reset_method(reset)


    def set_rhs(self, rhs=None, reset=True):
        """Set rhs functions. rhs may either be a dict specifying multiple functions related
        to the rhs, (e.g. {'rhs': f, 'rhs_jac': g}), or a callable, in which case it will be
        assumed to be the standard rhs function.
        """

        if rhs is None:
            rhs = {'rhs': None}

        if callable(rhs):
            rhs = {'rhs': rhs}

        if 'rhs' not in rhs:
            raise Exception('ODE_Method requires at minimum a specification of an rhs function.')

        self.rhs = self._state_type_converter.transform_rhs_funcs(rhs)

        self._reset_method(reset)


    """
    Functions to implement in concrete subclasses
    """

    @abstractmethod
    def integrate(self, tf):
        """Integrate up to a time tf.

        Args:
            tf (float): time to integrate up to
        """
        pass

    def _reset_method(self, reset=True):
        """Reset any parameters of internal numerical solving method, e.g. delete persistent memory
        for multi-step methods.

        Args:
            reset (bool): Whether or not to reset method
        """
        pass

    def set_options(self, solver_options):
        pass


class ScipyODE(ODE_Method):
    """Method wrapper for scipy.integrate.solve_ivp

    To use:
        - Specify a method acceptable by scipy.integrate.solve_ivp in solver_options using key
          'method'
        - Options for solve_ivp in the form of Keyword arguments may also be passed as a dict
          with key 'scipy_options' in solver_options
    """

    method_spec = {'inner_state_spec': {'type': 'array', 'ndim': 1}}

    def integrate(self, tf):
        """Integrate up to a time tf.
        """
        t0 = self.t
        y0 = self._y
        rhs = self.rhs.get('rhs')

        results = solve_ivp(rhs, (t0, tf), y0, method=self._scipy_method, **self._scipy_options)

        self._y = results.y[:, -1]
        self._t = results.t[-1]

    def set_options(self, solver_options):
        """Only option is max step size
        """
        if 'method' not in solver_options:
            raise Exception("""ScipyODE requires a 'method' key in solver_options with value a
                            method string acceptable by scipy.integrate.solve_ivp.""")
        self._scipy_method = solver_options.get('method')

        self._scipy_options = solver_options.get('scipy_options', {})


class RK4(ODE_Method):
    """
    Simple single-step RK4 solver
    """

    def integrate(self, tf):
        """Integrate up to a time tf.
        """

        delta_t = tf - self.t
        steps = int((delta_t // self._max_dt) + 1)
        h = delta_t / steps
        for k in range(steps):
            self._integration_step(h)

    def _integration_step(self, h):
        """Integration step for RK4
        """
        y0 = self._y
        t0 = self._t
        rhs = self.rhs.get('rhs')

        k1 = rhs(t0, y0)
        t_mid = t0 + (h / 2)
        k2 = rhs(t_mid, y0 + (h * k1 / 2))
        k3 = rhs(t_mid, y0 + (h * k2 / 2))
        t_end = t0 + h
        k4 = rhs(t_end, y0 + h * k3)
        self._y = y0 + (1. / 6) * h * (k1 + (2 * k2) + (2 * k3) + k4)
        self._t = t_end

    def set_options(self, solver_options):
        """Only option is max step size
        """
        if 'max_dt' not in solver_options:
            raise Exception('Solver requires max_dt setting')
        self._max_dt = solver_options['max_dt']


def method_from_string(method_str):
    """Factory function that returns a method specified by a string, along with any additional
    required options.

    Args:
        method_str (str): string specifying method

    Returns:
        (method, additional_options): method is the ODE_Method object, and additional_options
                                      is a dict containing any necessary options for that solver
    """

    method_dict = {'RK4': RK4}

    if method_str in method_dict:
        return method_dict.get(method_str), {}

    if 'scipy-' in method_str:
        return ScipyODE, {'method': method_str[6:]}
