# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=arguments-differ, missing-return-type-doc
"""
Qiskit Aer pulse simulator backend.
"""

import logging
from copy import deepcopy
from warnings import warn
from numpy import inf
from qiskit.providers.models import BackendConfiguration, PulseDefaults
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.aer.pulse.controllers.pulse_controller import pulse_controller
from qiskit.providers.aer.pulse.system_models.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.version import __version__

logger = logging.getLogger(__name__)

DEFAULT_CONFIGURATION = {
    'backend_name': 'pulse_simulator',
    'backend_version': __version__,
    'n_qubits': 20,
    'coupling_map': None,
    'url': 'https://github.com/Qiskit/qiskit-aer',
    'simulator': True,
    'meas_levels': [1, 2],
    'local': True,
    'conditional': True,
    'open_pulse': True,
    'memory': False,
    'max_shots': int(1e6),
    'description': 'A Pulse-based Hamiltonian simulator for Pulse Qobj files',
    'gates': [],
    'basis_gates': []
}


class PulseSimulator(AerBackend):
    r"""Pulse schedule simulator backend.

    The ``PulseSimulator`` simulates continuous time Hamiltonian dynamics of a quantum system,
    with controls specified by pulse :class:`~qiskit.Schedule` objects, and the model of the
    physical system specified by :class:`~qiskit.providers.aer.pulse.PulseSystemModel` objects.
    Results are returned in the same format as when jobs are submitted to actual devices.

    **Example**

    To use the simulator, first :func:`~qiskit.assemble` a :class:`PulseQobj` object
    from a list of pulse :class:`~qiskit.Schedule` objects, using ``backend=PulseSimulator()``.
    Call the simulator with the :class:`PulseQobj` and a
    :class:`~qiskit.providers.aer.pulse.PulseSystemModel` object representing the physical system.

    .. code-block:: python

        backend_sim = qiskit.providers.aer.PulseSimulator()

        # Assemble schedules using PulseSimulator as the backend
        pulse_qobj = assemble(schedules, backend=backend_sim)

        # Run simulation on a PulseSystemModel object
        results = backend_sim.run(pulse_qobj, system_model)

    **Supported PulseQobj parameters**

    * ``qubit_lo_freq``: Local oscillator frequencies for each :class:`DriveChannel`.
      Defaults to either the value given in the
      :class:`~qiskit.providers.aer.pulse.PulseSystemModel`, or is calculated directly
      from the Hamiltonian.
    * ``meas_level``: Type of desired measurement output, in ``[1, 2]``.
      ``1`` gives complex numbers (IQ values), and ``2`` gives discriminated states ``|0>`` and
      ``|1>``. Defaults to ``2``.
    * ``meas_return``: Measurement type, ``'single'`` or ``'avg'``. Defaults to ``'avg'``.
    * ``shots``: Number of shots per experiment. Defaults to ``1024``.


    **Simulation details**

    The simulator uses the ``zvode`` differential equation solver method through ``scipy``.
    Simulation is performed in the rotating frame of the diagonal of the drift Hamiltonian
    contained in the :class:`~qiskit.providers.aer.pulse.PulseSystemModel`. Measurements
    are performed in the `dressed basis` of the drift Hamiltonian.

    **Other options**

    :meth:`PulseSimulator.run` takes an additional ``dict`` argument ``backend_options`` for
    customization. Accepted keys:

    * ``'solver_options'``: A ``dict`` for solver options. Accepted keys
      are ``'atol'``, ``'rtol'``, ``'nsteps'``, ``'max_step'``, ``'num_cpus'``, ``'norm_tol'``,
      and ``'norm_steps'``.
    """
    def __init__(self,
                 configuration=None,
                 properties=None,
                 defaults=None,
                 provider=None,
                 system_model=None,
                 **backend_options):

        if configuration is None:
            configuration = BackendConfiguration.from_dict(
                DEFAULT_CONFIGURATION)
        else:
            configuration = deepcopy(configuration)
            configuration.simulator = True
            _set_config_meas_level(configuration, configuration.meas_levels)

        if defaults is None:
            defaults = PulseDefaults(qubit_freq_est=[inf],
                                     meas_freq_est=[inf],
                                     buffer=0,
                                     cmd_def=[],
                                     pulse_library=[])
        else:
            defaults = deepcopy(defaults)

        if properties is not None:
            properties = deepcopy(properties)

        # set up system model
        subsystem_list = backend_options.get('subsystem_list', None)
        if system_model is None:
            if hasattr(configuration, 'hamiltonian'):
                system_model = PulseSystemModel.from_config_and_defaults(configuration,
                                                                         defaults,
                                                                         subsystem_list)
        else:
            if hasattr(configuration, 'hamiltonian'):
                warn('''System model specified both as kwarg and in configuration,
                        defaulting to kwarg.''')

        self._system_model = system_model

        super().__init__(configuration,
                         properties=properties,
                         defaults=defaults,
                         provider=provider,
                         backend_options=backend_options)

    @classmethod
    def from_backend(cls, backend, **options):
        """Initialize simulator from backend."""
        configuration = backend.configuration()
        defaults = backend.defaults()

        backend_name = 'pulse_simulator({})'.format(configuration.backend_name)
        description = 'A Pulse-based simulator configured from the backend: '
        description += configuration.backend_name

        sim = cls(configuration=configuration,
                  defaults=defaults,
                  system_model=None,
                  backend_name=backend_name,
                  description=description,
                  **options)
        return sim

    def _execute(self, qobj, run_config):
        """Execute a qobj on the backend.

        Args:
            qobj (QasmQobj): simulator input.
            run_config (dict): run config for overriding Qobj config.

        Returns:
            dict: return a dictionary of results.
        """
        # preserve overriding of system model at run time
        system_model = run_config.get('system_model', self._system_model)
        return pulse_controller(qobj, system_model, run_config)

    def _set_option(self, key, value):

        if hasattr(self._configuration, key):
            if key == 'meas_levels':
                _set_config_meas_level(self._configuration, value)
            else:
                setattr(self._configuration, key, value)
        elif hasattr(self._defaults, key):
            setattr(self._defaults, key, value)

        elif hasattr(self._properties, key):
            setattr(self._properties, key, value)
        elif key == 'system_model':
            if hasattr(self._configuration, 'hamiltonian'):
                warn('''Specification of system model with configuration containing a hamiltonian
                        may result in backend inconsistencies.''')

            self._system_model = value
        else:
            self._options[key] = value


def _set_config_meas_level(configuration, meas_levels):
    """Function for setting meas_levels in a pulse simulator configuration."""
    meas_levels = deepcopy(meas_levels)
    if 0 in meas_levels:
        warn('Measurement level 0 not supported in pulse simulator.')
        meas_levels.remove(0)

    configuration.meas_levels = meas_levels
