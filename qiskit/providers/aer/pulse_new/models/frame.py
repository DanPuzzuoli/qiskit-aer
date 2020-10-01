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

from abc import ABC, abstractmethod
from typing import Callable, Union, List, Optional, Tuple
import numpy as np

from qiskit.quantum_info.operators import Operator

class BaseFrame(ABC):
    """Abstract interface for functionality for entering a constant rotating
    frame specified by an anti-Hermitian matrix F, and helper functions for
    doing computations in a basis in which F is diagonal.

    Assumption:
        - F is anti-Hermitian
    """

    @proeprty
    @abstractmethod
    def frame_operator(self) -> np.array:
        """The original frame operator."""

    @property
    @abstractmethod
    def frame_diag(self) -> np.array:
        """Diagonal of the frame operator."""

    @property
    @abstractmethod
    def frame_basis(self) -> np.array:
        """Array containing diagonalizing unitary."""

    @property
    @abstractmethod
    def frame_basis_adjoint(self) -> np.array:
        """Adjoint of the diagonalizing unitary."""

    def state_into_frame_basis(self, y: np.array) -> np.array:
        """Transform y into the frame basis.

        Args:
            y: the state
        Returns:
            np.array: the state in the frame basis
        """
        return self.frame_basis_adjoint @ y

    def state_out_of_frame_basis(self, y: np.array) -> np.array:
        """Transform y out of the frame basis.

        Args:
            y: the state
        Returns:
            np.array: the state in the frame basis
        """
        return self.frame_basis @ y

    def operator_into_frame_basis(self,
                                  op: Union[Operator, np.array]) -> np.array:
        """Transform operator into frame basis.

        Args:
            op: the operator.
        Returns:
            np.array: the operator in the frame basis
        """

        return self.frame_basis_adjoint @ to_array(op) @ self.frame_basis

    def operator_out_of_frame_basis(self,
                                    op: Union[Operator, np.array]) -> np.array:
        """Transform operator into frame basis.

        Args:
            op: the operator.
        Returns:
            np.array: the operator in the frame basis
        """

        return self.frame_basis @ to_array(op) @ self.frame_basis_adjoint

    def operators_into_frame_basis(self,
                                   operators: Union[List[Operator], np.array]) -> np.array:
        """Given a list of operators, perform a change of basis on all into
        the frame basis, and return as a 3d array.

        Args:
            operators: list of operators
        """

        return np.array([self.operator_into_frame_basis(o) for o in operators])

    @abstractmethod
    def state_into_frame(t: float,
                         y: np.array,
                         y_in_frame_basis: Optional[bool] = False,
                         return_in_frame_basis: Optional[bool] = False):
        """Take a state into the frame, i.e. return exp(-Ft) @ y.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """

        out = y

        # if not in frame basis convert it
        if not y_in_frame_basis:
            out = self.state_into_frame_basis(out)

        # go into the frame
        out = np.diag(np.exp(- t * self.frame_diag)) @ out

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.state_out_of_frame_basis(out)

        return out

    def state_out_of_frame(self,
                           t: float,
                           y: np.array,
                           y_in_frame_basis: Optional[bool] = False,
                           return_in_frame_basis: Optional[bool] = False):
        """Take a state out of the frame, i.e. return exp(Ft) @ y.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        return self.state_into_frame(-t, y,
                                     y_in_frame_basis,
                                     return_in_frame_basis)

    def frame_conjugate_operator(self,
                                 t: float,
                                 operator: Union[Operator, np.array],
                                 operator_in_frame_basis: Optional[bool] = False,
                                 return_in_frame_basis: Optional[bool] = False):
        """Return exp(Ft) @ operator @ exp(-Ft)

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        out = to_array(operator)

        # if not in frame basis convert it
        if not operator_in_frame_basis:
            out = self.operator_into_frame_basis(out)

        # go into the frame
        trans_diag = np.exp(- t * self.frame_diag)
        # assumption that F is anti-Hermitian implies conjugation of
        # diagonal gives inversion
        out = np.diag(trans_diag) @ out @ np.diag(trans_diag.conj())

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.operator_out_of_frame_basis(out)

        return out

    def operator_into_frame(self,
                            t: float,
                            operator: Union[Operator, np.array],
                            operator_in_frame_basis: Optional[bool] = False,
                            return_in_frame_basis: Optional[bool] = False):
        """Take an operator into the frame, i.e. return
        exp(Ft) @ operator @ exp(-Ft) - F.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """

        out = to_array(operator)

        # if not in frame basis convert it
        if not operator_in_frame_basis:
            out = self.operator_into_frame_basis(out)

        # go into the frame
        trans_diag = np.exp(- t * self.frame_diag)
        # assumption that F is anti-Hermitian implies conjugation of
        # diagonal gives inversion
        out = np.diag(trans_diag) @ out @ np.diag(trans_diag.conj())
        out = out - np.diag(self.frame_diag)

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.operator_out_of_frame_basis(out)

        return out

    def operator_out_of_frame(self,
                              t: float,
                              operator: Union[Operator, np.array],
                              operator_in_frame_basis: Optional[bool] = False,
                              return_in_frame_basis: Optional[bool] = False):
        """Take an operator out of the frame, i.e. return
        exp(-Ft) @ operator @ exp(Ft) + F.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """

        out = to_array(operator)

        # if not in frame basis convert it
        if not operator_in_frame_basis:
            out = self.operator_into_frame_basis(out)

        # go into the frame
        trans_diag = np.exp(t * self.frame_diag)
        # assumption that F is anti-Hermitian implies conjugation of
        # diagonal gives inversion
        out = np.diag(trans_diag) @ out @ np.diag(trans_diag.conj())
        out = out + np.diag(self.frame_diag)

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.operator_out_of_frame_basis(out)

        return out


    def _get_rotating_freq_and_cutoff_array(self,
                                            carrier_freqs: np.array,
                                            cutoff_freq: Optional[float] = None)
                                            -> Tuple[np.array, Union[None, np.array]]:
        """Get frequency and cutoff arrays in basis in which F is diagonal.
        """
        # create difference matrix for diagonal elements
        dim = len(self.frame_diag)
        D_diff = np.ones((dim, dim)) * self.frame_diag
        D_diff = D_diff - D_diff.transpose()

        # set up matrix encoding frequencies
        im_angular_freqs = 1j * 2 * np.pi * carrier_freqs
        freq_array = np.array([w + D_diff for w in im_angular_freqs])

        # set up cutoff frequency matrix - i.e. same shape as freq_array - with
        # each entry a 1 if the corresponding entry of freq_array has a
        # frequency below the cutoff, and 0 otherwise
        cutoff_array = None
        if cutoff_freq is not None:
            cutoff_array = ((np.abs(freq_array.imag) / (2 * np.pi))
                                    < cutoff_freq).astype(int)

        return freq_array, cutoff_array

    def evaluate_operator_linear_combo(self,
                                       t: float,
                                       coefficients: np.array,
                                       operators_in_frame_basis: np.array,
                                       freq_array: np.array,
                                       cutoff_array: Optional[np.array] = None,
                                       in_frame_basis: Optional[bool] = False):
        """Evaluate the "canonical" decomposition.

        Explain this!!
        """
        # first evaluate the unconjugated coefficients for each matrix element,
        # given by the coefficient for the full matrix multiplied by the
        # exponentiated frequency term for each entry
        Q = (coefficients[:, np.newaxis, np.newaxis] * np.exp(freq_array * t))

        # apply cutoff if present
        if cutoff_array is not None:
           Q = cutoff_array * Q

        # multiplying the operators by the average of the "unconjugated" and
        # "conjugated" coefficients
        op_list = (0.5 * (Q + Q.conj().transpose(0, 2, 1)) *
                   operators_in_frame_basis)

        # sum the operators and subtract the frame operator
        op_in_frame_basis = np.sum(op_list, axis=0) - np.diag(self.frame_diag)

        if in_frame_basis:
            return op_in_frame_basis
        else:
            return (self.frame_basis @ op_in_frame_basis @
                    self.frame_basis_adjoint)


# type handling
def to_array(op):
    if isinstance(op, Operator):
        return op.data
    return op
