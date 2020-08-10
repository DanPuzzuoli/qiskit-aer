# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
PulseSimulator Integration Tests
"""

import sys
import unittest
from test.terra import common

from qiskit.test.mock.backends.athens.fake_athens import FakeAthens

from qiskit.providers.aer.backends import PulseSimulator

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

    def test_bare_instance(self):
        """Test behaviour of PulseSimulator instance with no configuration or defaults."""

        pass

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

        # qubit_freq_est
        athens_attr = athens_backend.defaults().qubit_freq_est
        sim_attr = athens_sim.defaults().qubit_freq_est
        model_attr = athens_sim._system_model._qubit_freq_est
        self.assertTrue(sim_attr == athens_attr and model_attr == athens_attr)

        # meas_freq_est
        athens_attr = athens_backend.defaults().meas_freq_est
        sim_attr = athens_sim.defaults().meas_freq_est
        model_attr = athens_sim._system_model._meas_freq_est
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



if __name__ == '__main__':
    unittest.main()
