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
    """Hamiltonian model

    ********Add comments about frames being handled assuming the structure
    of the Schrodinger Equation
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

        return 1j * self.frame.generator_into_frame(time,
                                                    -1j * op_combo,
                                                    operator_in_frame_basis=True,
                                                    return_in_frame_basis=in_frame_basis)
