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
from .signals import VectorSignal, Constant, Signal
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

        op_combo = self._evaluate_in_frame_basis_with_cutoffs(sig_vals)

        op_to_add_in_fb = None
        if self.frame.frame_operator is not None:
            op_to_add_in_fb = 1j * np.diag(self.frame.frame_diag)


        return self.frame._conjugate_and_add(time,
                                             op_combo,
                                             op_to_add_in_fb=op_to_add_in_fb,
                                             operator_in_frame_basis=True,
                                             return_in_frame_basis=in_frame_basis)

class QuantumSystemModel:
    """A model of a quantum system.

    For now, contains:
        - a hamiltonian model
        - a list of noise operators (optional)
        - a list of noise coefficients (optional)

    At the moment not quite sure how a user may want to interact with this.
    Perhaps it should also be an :class:`OperatorModel` subclass?
    """

    def __init__(self,
                 hamiltonian,
                 noise_operators=None,
                 noise_signals=None):

        self.hamiltonian = hamiltonian
        self.noise_operators = noise_operators

        if noise_signals is None and noise_operators is not None:
            noise_signals = [Constant(1.) for _ in noise_operators]

        self.noise_signals = noise_signals

    @property
    def vectorized_lindblad_generator(self):
        """Get the `OperatorModel` representing the vectorized Lindblad
        equation.

        I.e. the Lindblad equation is:
        .. math::
            \dot{\rho}(t) = -i[H(t), \rho(t)] + \mathcal{D}(t)(\rho(t)),

        where :math:`\mathcal{D}(t)` is the dissipator portion of the equation,
        given by

        .. math::
            \mathcal{D}(t)(\rho(t)) = \sum_j \gamma_j(t) L_j \rho L_j^\dagger - \frac{1}{2} \{L_j^\dagger L_j, \rho\},

        with :math:`[\cdot, \cdot]` and :math:`\{\cdot, \cdot\}` the
        matrix commutator and anti-commutator, respectively.


        """

        # combine operators
        vec_ham_ops = -1j * vec_commutator(to_array(self.hamiltonian._operators))

        full_operators = None
        if self.noise_operators is not None:
            vec_diss_ops = vec_dissipator(to_array(self.noise_operators))
            full_operators = np.append(vec_ham_ops, vec_diss_ops, axis=0)
        else:
            full_operators = vec_ham_ops

        # combine signals
        # it will take some thought to make this nice but for now I'll just
        # put it together quickly
        ham_sigs = self.hamiltonian.signals

        full_signals = None
        if self.noise_operators is not None:
            noise_sigs = None
            if isinstance(self.noise_signals, VectorSignal):
                noise_sigs = self.noise_signals
            elif isinstance(self.noise_signals, list):
                noise_sigs = VectorSignal.from_signal_list(self.noise_signals)

            full_envelope = lambda t: np.append(ham_sigs.envelope(t),
                                                noise_sigs.envelope(t))
            full_carrier_freqs = np.append(ham_sigs.carrier_freqs,
                                          noise_sigs.carrier_freqs)

            full_drift_array = np.append(ham_sigs.drift_array,
                                         noise_sigs.drift_array)

            full_signals = VectorSignal(envelope=full_envelope,
                                        carrier_freqs=full_carrier_freqs,
                                        drift_array=full_drift_array)
        else:
            full_signals = ham_sigs


        return OperatorModel(operators=full_operators,
                             signals=full_signals)





def vec_commutator(A):
    """Linear algebraic vectorization of the linear map X -> [A, X]
    in row-stacking convention.

    Note: this function is also "vectorized" in the programming sense.
    """
    iden = np.eye(A.shape[-1])
    axes = list(range(A.ndim))
    axes[-1] = axes[-2]
    axes[-2] += 1
    return np.kron(A, iden) - np.kron(iden, A.transpose(axes))

def vec_dissipator(L):
    """ Linear algebraic vectorization of the linear map
    X -> L X L^\dagger - 0.5 * (L^\dagger L X + X L^\dagger L).

    Note: this function is also "vectorized" in the programming sense.
    """
    iden = np.eye(L.shape[-1])
    axes = list(range(L.ndim))

    axes[-1] = axes[-2]
    axes[-2] += 1
    Lconj = L.conj()
    LdagL = Lconj.transpose(axes) @ L

    # Note: below uses that, if L.ndim==2, LdagL.transpose() == LdagL.conj()
    return np.kron(L, iden) @ np.kron(iden, Lconj) - 0.5 * (np.kron(LdagL, iden) +
            np.kron(iden, LdagL.conj()))


def to_array(op: Union[Operator, np.array, List[Operator], List[np.array]]):
    """Convert an operator, either specified as an `Operator` or an array
    to an array.

    Args:
        op: the operator to represent as an array.
    Returns:
        np.array: op as an array
    """
    if isinstance(op, list):
        return np.array([to_array(sub_op) for sub_op in op])

    if isinstance(op, Operator):
        return op.data
    return op
