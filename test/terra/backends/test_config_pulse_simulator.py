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

        pulse_sim = PulseSimulator()
        



if __name__ == '__main__':
    unittest.main()
