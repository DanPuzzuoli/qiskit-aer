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

from typing import Callable, Union, List, Optional
import numpy as np
from .signals import VectorSignal, Signal
from qiskit.quantum_info.operators import Operator
from .frame import Frame
from .operator_models import OperatorModel

class HamiltonianModel(OperatorModel):
    """A model of a Hamiltonian, i.e. a time-dependent operator of the form

    .. math::

        H(t) = \sum_{i=0}^{k-1} s_i(t) H_i,

    where :math:`H_i` are Hermitian operators, and the :math:`s_i(t)` are
    time-dependent functions represented by :class:`Signal` objects.

    Functionally this class behaves the same as :class:`OperatorModel`,
    with the following modifications:
        - The operators in the linear decomposition are verified to be
          Hermitian.
        - Frames are dealt with assuming the structure of the Schrodinger
          equation. I.e. Evaluating the Hamiltonian :math:`H(t)` in a
          frame :math:`F = -iH`, evaluates the expression
          :math:`e^{-tF}H(t)e^{tF} - H`. This is in contrast to
          the base class :class:`OperatorModel`, which would ordinarily
          evaluate :math:`e^{-tF}H(t)e^{tF} - F`.
    """

    def __init__(self,
                 operators: List[Operator],
                 signals: Optional[Union[VectorSignal, List[Signal]]] = None,
                 signal_mapping: Optional[Callable] = None,
                 frame: Optional[Union[Operator, np.array]] = None,
                 cutoff_freq: Optional[float] = None):

        for operator in operators:
            if np.linalg.norm((operator.adjoint() - operator).data) > 1e-10:
                raise Exception("""HamiltonianModel only accepts Hermitian
                                    operators.""")

        # allow OperatorModel to instantiate everything but frame
        super().__init__(operators=operators,
                         signals=signals,
                         signal_mapping=signal_mapping,
                         frame=frame,
                         cutoff_freq=cutoff_freq)


    @property
    def frame(self) -> Frame:
        """Return the frame."""
        return self._frame

    @frame.setter
    def frame(self, frame: Union[Operator, np.array, Frame]):
        """Set the frame. For a HamiltonianModel, we handle the following
        possibilities:

        - frame is None
        - frame is a Hermitian operator, in which case it is interpreted
          as entering the frame of a given Hamiltonian H (corresponding to
          frame operator F= -1j * H)
        - frame is anti-Hermitian, in which case H = 1j * F, and the same
          interpretation as previous point is used for H
        - frame is already a defined Frame object, in which case the
          previous point applies.
        """

        if frame is None:
            self._frame = Frame(None)
        else:
            if isinstance(frame, Frame):
                self._frame = frame
            else:
                if isinstance(frame, Operator):
                    frame = frame.data

                # if frame operator is Hermitian, assume it is meant to
                # be a Hamiltonian
                if np.linalg.norm(frame.conj().transpose() - frame) < 1e-12:
                    self._frame = Frame(-1j * frame)
                else:
                    self._frame = Frame(frame)

        self._reset_internal_ops()

    def evaluate(self, time: float, in_frame_basis: bool = False) -> np.array:
        """
        Note: evaluation of OperatorModel gives exp(-tF)@ G(t) @ exp(tF) - F,
        however in this case we have F = -i H0, and what we want is
        exp(-tF) @ H(t) @ exp(tF) - H0, where G(t) = -i H(t).

        To utilize the existing functions, we multiply the evaluated H(t)
        from evaluate_linear_combo by -1j, then after mapping into the frame
        undo this via multiplication by 1j.

        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal

        Returns:
            np.array: the evaluated model
        """

        if self.signals is None:
            raise Exception("""OperatorModel cannot be
                               evaluated without signals.""")

        sig_vals = self.signals.value(time)

        op_combo = self._evaluate_linear_combo(sig_vals)

        op_to_add_in_fb = None
        if self.frame.frame_operator is not None:
            op_to_add_in_fb = 1j * np.diag(self.frame.frame_diag)


        return self.frame._conjugate_and_add(time,
                                             op_combo,
                                             op_to_add_in_fb=op_to_add_in_fb,
                                             operator_in_frame_basis=True,
                                             return_in_frame_basis=in_frame_basis)
