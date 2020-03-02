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
from numpy import inf
from qiskit.providers.models import BackendConfiguration, PulseDefaults
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.aer.pulse.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.pulse.qobj.digest import digest_pulse_obj
from qiskit.providers.aer.pulse.solver.opsolve import opsolve
from qiskit.providers.aer.version import __version__

logger = logging.getLogger(__name__)

DEFAULT_CONFIGURATION = {
    'backend_name': 'pulse_simulator',
    'backend_version': __version__,
    'n_qubits': 20,
    'coupling_map': None,
    'url': 'https://github.com/Qiskit/qiskit-aer',
    'simulator': True,
    'meas_levels': [0, 1, 2],
    'local': True,
    'conditional': True,
    'open_pulse': True,
    'memory': False,
    'max_shots': int(1e6),
    'description': 'A pulse-based Hamiltonian simulator for Pulse Qobj files',
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

    def __init__(self, provider=None, **backend_options):

        # purpose of defaults is to pass assemble checks
        self._defaults = PulseDefaults(qubit_freq_est=[inf],
                                       meas_freq_est=[inf],
                                       buffer=0,
                                       cmd_def=[],
                                       pulse_library=[])
        super().__init__(BackendConfiguration.from_dict(DEFAULT_CONFIGURATION),
                         provider=provider,
                         backend_options=backend_options)

    # pylint: disable=arguments-differ
    def run(self, qobj, system_model=None,
            validate=False,
            backend_options=None,
            **kwargs):
        """Run a qobj on system_model.

        Args:
            qobj (PulseQobj): Qobj for pulse Schedules to run
            system_model (PulseSystemModel or None): Physical model to run simulation on
            validate (bool): Flag for validation checks
            backend_options (dict): Other options
            kwargs (any): optional additional backend options.

        Returns:
            Result: results of simulation
        """
        return super().run(qobj,
                           validate=validate,
                           backend_options=backend_options,
                           system_model=system_model, **kwargs)

    @classmethod
    def from_backend(cls, backend, **options):
        """Initialize simulator from backend."""
        configuration = backend.configuration()
        coupling_map = configuration.coupling_map
        backend_name = 'pulse_simulator({})'.format(configuration.backend_name)
        system_model = PulseSystemModel.from_backend(backend, subsystem_list=None)
        sim = cls(system_model=system_model,
                  coupling_map=coupling_map,
                  backend_name=backend_name,
                  **options)
        return sim

    def _execute(self, qobj, run_config):
        """Execute qobj"""
        system_model = run_config.get('system_model')
        openpulse_system = digest_pulse_obj(qobj, system_model, run_config)
        results = opsolve(openpulse_system)
        output = {}
        output['qobj_id'] = qobj.qobj_id
        output['results'] = results
        output['success'] = True
        return output

    def defaults(self):
        """Return defaults.

        Returns:
            PulseDefaults: object for passing assemble.
        """
        return self._defaults
