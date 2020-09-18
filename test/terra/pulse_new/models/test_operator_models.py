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
"""tests for operator_models.py"""

import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.pulse_new.models.operator_models import FrameFreqHelper, OperatorModel, vector_apply_diag_frame


class Test_FrameFreqHelper(unittest.TestCase):

    def setUp(self):
        self.X = Operator(np.array([[0., 1.], [1., 0.]], dtype=complex))
        self.Y = Operator(np.array([[0., -1j], [1j, 0.]], dtype=complex))
        self.Z = Operator(np.array([[1., 0.], [0., -1.]], dtype=complex))

    def test_evaluate_no_frame(self):
        """Test FrameFreqHelper.evaluate with no frame or cutoff."""

        operators = [self.X, self.Y, self.Z]
        carrier_freqs = np.array([1., 2., 3.])

        ffhelper = FrameFreqHelper(operators, carrier_freqs)

        t = 0.123
        coeffs = np.array([1., 1j, 1 + 1j])

        out = ffhelper.evaluate(t, coeffs)
        sig_vals = np.real(coeffs * np.exp(1j * 2 * np.pi * carrier_freqs * t))
        ops_as_arrays = np.array([op.data for op in operators])
        expected_out = np.tensordot(sig_vals, ops_as_arrays, axes=1)

        self.assertAlmostEqual(out, expected_out)

        t = 0.123 * np.pi
        coeffs = np.array([4.131, 3.23, 2.1 + 3.1j])

        out = ffhelper.evaluate(t, coeffs)
        sig_vals = np.real(coeffs * np.exp(1j * 2 * np.pi * carrier_freqs * t))
        ops_as_arrays = np.array([op.data for op in operators])
        expected_out = np.tensordot(sig_vals, ops_as_arrays, axes=1)

        self.assertAlmostEqual(out, expected_out)

    def test_state_transformations_no_frame(self):
        """Test frame transformations with no frame."""

        operators = [self.X]
        carrier_freqs = np.array([1.])

        ffhelper = FrameFreqHelper(operators, carrier_freqs)

        t = 0.123
        y = np.array([1., 1j])
        out = ffhelper.state_into_frame(t, y)
        self.assertAlmostEqual(out, y)
        out = ffhelper.state_out_of_frame(t, y)
        self.assertAlmostEqual(out, y)

        t = 100.12498
        y = np.eye(2)
        out = ffhelper.state_into_frame(t, y)
        self.assertAlmostEqual(out, y)
        out = ffhelper.state_out_of_frame(t, y)
        self.assertAlmostEqual(out, y)

    def test_internal_helper_mats_no_cutoff(self):
        """Test internal setup steps for helper matrices with no freq cutoff"""

        # no cutoff with already diagonal frame
        frame_op = -1j * np.pi * np.array([1., -1.])
        operators = [self.X, self.X, self.X]
        carrier_freqs = np.array([1., 2., 3.])

        helper = FrameFreqHelper(operators, carrier_freqs, frame_op)

        D_diff = -1j * np.pi * np.array([[0, -2.], [2., 0.]])
        im_freqs = 1j * 2 * np.pi * carrier_freqs
        S_expect = np.array([w + D_diff for w in im_freqs])
        M_expect = None

        self.assertTrue(helper._M_cutoff == M_expect)
        self.assertAlmostEqual(helper._S, S_expect)

        # same test but with frame given as a 2d array
        # in this case diagonalization will occur, causing eigenvalues to
        # be sorted in ascending order
        frame_op = -1j * np.pi * np.array([[-1., 0], [0, 1.]])
        operators = [self.X, self.X, self.X]
        carrier_freqs = np.array([1., 2., 3.])

        helper = FrameFreqHelper(operators, carrier_freqs, frame_op)

        D_diff = -1j * np.pi * np.array([[0, 2.], [-2., 0.]])
        im_freqs = 1j * 2 * np.pi * carrier_freqs
        S_expect = np.array([w + D_diff for w in im_freqs])
        M_expect = None

        self.assertTrue(helper._M_cutoff == M_expect)
        self.assertAlmostEqual(helper._S, S_expect)

    def test_internal_helper_mats_with_cutoff(self):
        """Test internal setup steps for helper matrices with rwa freq cutoff"""

        # cutoff test
        frame_op = -1j * np.pi * np.array([1., -1.])
        operators = [self.X, self.X, self.X]
        carrier_freqs = np.array([1., 2., 3.])
        cutoff_freq = 3.

        helper = FrameFreqHelper(operators,
                                 carrier_freqs,
                                 frame_op,
                                 cutoff_freq)

        D_diff = -1j * np.pi * np.array([[0, -2.], [2., 0.]])
        im_freqs = 1j * 2 * np.pi * carrier_freqs
        S_expect = np.array([w + D_diff for w in im_freqs])
        M_expect = np.array([[[1, 1],
                              [1, 1]],
                             [[1, 0],
                              [1, 1]],
                             [[0, 0],
                              [1, 0]]
                             ])

        self.assertAlmostEqual(helper._M_cutoff, M_expect)
        self.assertAlmostEqual(helper._S, S_expect)

        # same test with lower cutoff
        cutoff_freq = 2.

        helper = FrameFreqHelper(operators,
                                 carrier_freqs,
                                 frame_op,
                                 cutoff_freq)

        D_diff = -1j * np.pi * np.array([[0, -2.], [2., 0.]])
        im_freqs = 1j * 2 * np.pi * carrier_freqs
        S_expect = np.array([w + D_diff for w in im_freqs])
        M_expect = np.array([[[1, 0],
                              [1, 1]],
                             [[0, 0],
                              [1, 0]],
                             [[0, 0],
                              [0, 0]]
                             ])

        self.assertAlmostEqual(helper._M_cutoff, M_expect)
        self.assertAlmostEqual(helper._S, S_expect)

    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)
