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
"""
Configurable PulseSimulator Tests
"""

import sys
import unittest
import warnings
import numpy as np
from test.terra import common

from qiskit.test.mock.backends.armonk.fake_armonk import FakeArmonk
from qiskit.test.mock.backends.athens.fake_athens import FakeAthens

from qiskit.providers.aer.backends import PulseSimulator
from qiskit.pulse import (Schedule, Play, ShiftPhase, SetPhase, Delay, Acquire,
                          Waveform, DriveChannel, ControlChannel,
                          AcquireChannel, MemorySlot)
from qiskit.providers.aer.aererror import AerError

from qiskit.compiler import assemble
from qiskit.providers.aer.pulse.system_models.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.pulse.system_models.hamiltonian_model import HamiltonianModel
from qiskit.providers.models.backendconfiguration import UchannelLO


class TestConfigPulseSimulator(common.QiskitAerTestCase):
    r"""PulseSimulator tests."""
    def setUp(self):
        """ Set configuration settings for pulse simulator
        WARNING: We do not support Python 3.5 because the digest algorithm relies on dictionary insertion order.
        This "feature" was introduced later on Python 3.6 and there's no official support for OrderedDict in the C API so
        Python 3.5 support has been disabled while looking for a propper fix.
        """
        if sys.version_info.major == 3 and sys.version_info.minor == 5:
            self.skipTest("We don't support Python 3.5 for Pulse simulator")

    def test_from_backend(self):
        """Test that configuration, defaults, and properties are correclty imported."""

        athens_backend = FakeAthens()
        athens_sim = PulseSimulator.from_backend(athens_backend)
        import pdb; pdb.set_trace()
        self.assertEqual(athens_backend.properties(), athens_sim.properties())
        #self.assertEqual(athens_backend.configuration(), athens_sim.configuration())
        self.assertEqual(athens_backend.defaults(), athens_sim.defaults())



    def test_from_backend_system_model(self):
        """Test that the system model is correctly imported from the backend."""

        athens_backend = FakeAthens()
        athens_sim = PulseSimulator.from_backend(athens_backend)

        # u channel lo
        athens_attr = athens_backend.configuration().u_channel_lo
        sim_attr = athens_sim.configuration().u_channel_lo
        model_attr = athens_sim._system_model.u_channel_lo
        self.assertTrue(sim_attr == athens_attr and model_attr == athens_attr)

        # dt
        athens_attr = athens_backend.configuration().dt
        sim_attr = athens_sim.configuration().dt
        model_attr = athens_sim._system_model.dt
        self.assertTrue(sim_attr == athens_attr and model_attr == athens_attr)

    def test_set_system_model_options(self):
        """Test setting of options that need to be changed in multiple places."""

        athens_backend = FakeAthens()
        athens_sim = PulseSimulator.from_backend(athens_backend)

        # u channel lo
        set_attr = [[UchannelLO(0, 1.0 + 0.0j)]]
        athens_sim.set_options(u_channel_lo=set_attr)
        sim_attr = athens_sim.configuration().u_channel_lo
        model_attr = athens_sim._system_model.u_channel_lo
        self.assertTrue(sim_attr == set_attr and model_attr == set_attr)

        # dt
        set_attr = 5.
        athens_sim.set_options(dt=set_attr)
        sim_attr = athens_sim.configuration().dt
        model_attr = athens_sim._system_model.dt
        self.assertTrue(sim_attr == set_attr and model_attr == set_attr)

        # qubit_freq_est
        set_attr = [5.]
        athens_sim.set_options(qubit_freq_est=set_attr)
        sim_attr = athens_sim.defaults().qubit_freq_est
        model_attr = athens_sim._system_model._qubit_freq_est
        self.assertTrue(sim_attr == set_attr and model_attr == set_attr)

        # meas_freq_est
        set_attr = [5.]
        athens_sim.set_options(meas_freq_est=set_attr)
        sim_attr = athens_sim.defaults().meas_freq_est
        model_attr = athens_sim._system_model._meas_freq_est
        self.assertTrue(sim_attr == set_attr and model_attr == set_attr)

    def test_set_meas_levels(self):
        """Test setting of meas_levels."""

        athens_backend = FakeAthens()
        athens_sim = PulseSimulator.from_backend(athens_backend)

        # test that a warning is thrown when meas_level 0 is attempted to be set
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            athens_sim.set_options(meas_levels=[0,1,2])

            self.assertEqual(len(w), 1)
            self.assertTrue('Measurement level 0 not supported' in str(w[-1].message))
            self.assertEqual(athens_sim.configuration().meas_levels, [1, 2])

        self.assertTrue(athens_sim.configuration().meas_levels == [1, 2])

        athens_sim.set_options(meas_levels=[2])
        self.assertTrue(athens_sim.configuration().meas_levels == [2])

    def test_set_system_model_from_backend(self):
        """Test setting system model when constructing from backend."""

        armonk_backend = FakeArmonk()
        system_model = self._system_model_1Q()

        # these are 1q systems so this doesn't make sense but can still be used to test
        system_model.u_channel_lo = [[UchannelLO(0, 1.0 + 0.0j)]]

        armonk_sim = None

        # construct backend and catch warning
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            armonk_sim = PulseSimulator.from_backend(backend=armonk_backend,
                                                     system_model=system_model)

            self.assertEqual(len(w), 1)
            self.assertTrue('inconsistencies' in str(w[-1].message))

        # check that system model properties have been imported
        self.assertEqual(armonk_sim.configuration().dt, system_model.dt)
        self.assertEqual(armonk_sim.configuration().u_channel_lo, system_model.u_channel_lo)
        self.assertEqual(armonk_sim.defaults().qubit_freq_est, system_model._qubit_freq_est)
        self.assertEqual(armonk_sim.defaults().meas_freq_est, system_model._meas_freq_est)

    def test_set_system_model_in_constructor(self):
        """Test setting system model when constructing."""

        system_model = self._system_model_1Q()

        # these are 1q systems so this doesn't make sense but can still be used to test
        system_model.u_channel_lo = [[UchannelLO(0, 1.0 + 0.0j)]]

        # construct directly
        test_sim = None
        # construct backend and verify no warnings
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            test_sim = PulseSimulator(system_model=system_model)

            self.assertEqual(len(w), 0)

        # check that system model properties have been imported
        self.assertEqual(test_sim.configuration().dt, system_model.dt)
        self.assertEqual(test_sim.configuration().u_channel_lo, system_model.u_channel_lo)
        self.assertEqual(test_sim.defaults().qubit_freq_est, system_model._qubit_freq_est)
        self.assertEqual(test_sim.defaults().meas_freq_est, system_model._meas_freq_est)

    def test_set_system_model_after_construction(self):
        """Test setting the system model after construction."""

        system_model = self._system_model_1Q()

        # these are 1q systems so this doesn't make sense but can still be used to test
        system_model.u_channel_lo = [[UchannelLO(0, 1.0 + 0.0j)]]

        # first test setting after construction with no hamiltonian
        test_sim = PulseSimulator()

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            test_sim.set_options(system_model=system_model)
            self.assertEqual(len(w), 0)

        # check that system model properties have been imported
        self.assertEqual(test_sim._system_model, system_model)
        self.assertEqual(test_sim.configuration().dt, system_model.dt)
        self.assertEqual(test_sim.configuration().u_channel_lo, system_model.u_channel_lo)
        self.assertEqual(test_sim.defaults().qubit_freq_est, system_model._qubit_freq_est)
        self.assertEqual(test_sim.defaults().meas_freq_est, system_model._meas_freq_est)

        # next, construct a pulse simulator with a config containing a Hamiltonian and observe
        # warnings
        armonk_backend = FakeArmonk()
        test_sim = PulseSimulator(configuration=armonk_backend.configuration())

        # add system model and verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            armonk_sim = test_sim.set_options(system_model=system_model)

            self.assertEqual(len(w), 1)
            self.assertTrue('inconsistencies' in str(w[-1].message))

        self.assertEqual(test_sim.configuration().dt, system_model.dt)
        self.assertEqual(test_sim.configuration().u_channel_lo, system_model.u_channel_lo)
        self.assertEqual(test_sim.defaults().qubit_freq_est, system_model._qubit_freq_est)
        self.assertEqual(test_sim.defaults().meas_freq_est, system_model._meas_freq_est)

    def test_validation_num_acquires(self):
        """Test that validation fails if 0 or >1 acquire is given in a schedule."""

        test_sim = PulseSimulator(system_model=self._system_model_1Q())

        qobj = assemble([self._1Q_invalid_sched()],
                        backend=test_sim,
                        meas_level=2,
                        qubit_lo_freq=[0.],
                        meas_return='single',
                        shots=256)

        try:
            test_sim.run(qobj)
        except AerError as error:
            import pdb; pdb.set_trace()
            print('wow')


    def _system_model_1Q(self, omega_0=5., r=0.02):
        """Constructs a standard model for a 1 qubit system.

        Args:
            omega_0 (float): qubit frequency
            r (float): drive strength

        Returns:
            PulseSystemModel: model for qubit system
        """

        hamiltonian = {}
        hamiltonian['h_str'] = [
            '2*np.pi*omega0*0.5*Z0', '2*np.pi*r*0.5*X0||D0'
        ]
        hamiltonian['vars'] = {'omega0': omega_0, 'r': r}
        hamiltonian['qub'] = {'0': 2}
        ham_model = HamiltonianModel.from_dict(hamiltonian)

        u_channel_lo = []
        subsystem_list = [0]
        dt = 1.

        return PulseSystemModel(hamiltonian=ham_model,
                                u_channel_lo=u_channel_lo,
                                subsystem_list=subsystem_list,
                                dt=dt)

    def _1Q_invalid_sched(self, num_acquires=2):
        """Creates a schedule with a variable number of acquires. num_acquires == 1 is a valid
        schedule, and both num_acquires == 0 and num_acquires > 1 should raise errors.

        Args:
            num_acquires (int): number of acquire instructions to include in the schedule

        Returns:
            schedule (pulse schedule):
        """

        total_samples = 100
        schedule = Schedule()
        schedule |= Play(Waveform(np.ones(total_samples)), DriveChannel(0))
        for _ in range(num_acquires):
            schedule |= Acquire(total_samples, AcquireChannel(0),
                                MemorySlot(0)) << schedule.duration
        return schedule


if __name__ == '__main__':
    unittest.main()
