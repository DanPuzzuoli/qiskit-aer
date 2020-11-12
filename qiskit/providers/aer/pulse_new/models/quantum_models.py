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
from .signals import VectorSignal, Constant, Signal, BaseSignal
from qiskit.quantum_info.operators import Operator
from .frame import Frame
from .operator_models import OperatorModel
from ..type_utils import vec_commutator, vec_dissipator, to_array

class HamiltonianModel(OperatorModel):
    """A model of a Hamiltonian, i.e. a time-dependent operator of the form

    .. math::

        H(t) = \sum_{i=0}^{k-1} s_i(t) H_i,

    where :math:`H_i` are Hermitian operators, and the :math:`s_i(t)` are
    time-dependent functions represented by :class:`Signal` objects.

    Currently the functionality of this class is as a subclass of
    :class:`OperatorModel`, with the following modifications:
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
        """Initialize, ensuring that the operators are Hermitian.

        Args:
            operators: list of Operator objects.
            signals: Specifiable as either a VectorSignal, a list of
                     Signal objects, or as the inputs to signal_mapping.
                     OperatorModel can be instantiated without specifying
                     signals, but it can not perform any actions without them.
            signal_mapping: a function returning either a
                            VectorSignal or a list of Signal objects.
            frame: Rotating frame operator. If specified with a 1d
                            array, it is interpreted as the diagonal of a
                            diagonal matrix.
            cutoff_freq: Frequency cutoff when evaluating the model.
        """

        # verify operators are Hermitian, and if so instantiate
        for operator in operators:
            if isinstance(operator, Operator):
                operator = operator.data

            if np.linalg.norm((operator.conj().transpose()
                                - operator).data) > 1e-10:
                raise Exception("""HamiltonianModel only accepts Hermitian
                                    operators.""")

        super().__init__(operators=operators,
                         signals=signals,
                         signal_mapping=signal_mapping,
                         frame=frame,
                         cutoff_freq=cutoff_freq)

    def evaluate(self, time: float, in_frame_basis: bool = False) -> np.array:
        """Evaluate the Hamiltonian at a given time.

        Note: This function from :class:`OperatorModel` needs to be overridden,
        due to frames for Hamiltonians being relative to the Schrodinger
        equation, rather than the Hamiltonian itself.
        See the class doc string for details.

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
            op_to_add_in_fb = -1j * np.diag(self.frame.frame_diag)


        return self.frame._conjugate_and_add(time,
                                             op_combo,
                                             op_to_add_in_fb=op_to_add_in_fb,
                                             operator_in_frame_basis=True,
                                             return_in_frame_basis=in_frame_basis)

class LindbladModel:
    """A model of a quantum system, consisting of a :class:`HamiltonianModel`
    and an optional description of dissipative dynamics.

    Dissipation terms are understood in terms of the Lindblad master
    equation:

    .. math::
        \dot{\rho}(t) = -i[H(t), \rho(t)] + \mathcal{D}(t)(\rho(t)),

    where :math:`\mathcal{D}(t)` is the dissipator portion of the equation,
    given by

    .. math::
        \mathcal{D}(t)(\rho(t)) = \sum_j \gamma_j(t) L_j \rho L_j^\dagger - \frac{1}{2} \{L_j^\dagger L_j, \rho\},

    with :math:`[\cdot, \cdot]` and :math:`\{\cdot, \cdot\}` the
    matrix commutator and anti-commutator, respectively. In the above:
        - :math:`H(t)` denotes the Hamiltonian,
        - :math:`L_j` denotes the :math:`j^{th}` noise, or Lindblad,
          operator, and
        - :math:`\gamma_j(t)` denotes the signal corresponding to the
          :math:`j^{th}` Lindblad operator.
    """

    def __init__(self,
                 hamiltonian: HamiltonianModel,
                 noise_operators: Optional[List[Operator]] = None,
                 noise_signals: Optional[Union[List[BaseSignal], VectorSignal]] = None):
        """Initialize. Noise parameters are optional. If `noise_operators`
        is specified but `noise_signals` is left as `None`, then internally
        sets the coefficient for each noise operator to `Constant(1.)`.

        Args:
            hamiltonian: the Hamiltonian.
            noise_operators: list of dissipation operators.
            noise_signals: list of time-dependent signals for the dissipation
                           operators.
        """

        self.hamiltonian = hamiltonian

        self.noise_operators = noise_operators

        if noise_signals is None and noise_operators is not None:
            noise_signals = [Constant(1.) for _ in noise_operators]

        self.noise_signals = noise_signals

    @property
    def vectorized_lindblad_generator(self):
        """Get the :class:`OperatorModel` representing the vectorized Lindblad
        equation (described in the class doc string), in column-stacking
        convention.

        In column stacking convention, the map :math:`X \mapsto -i[H(t), X]`
        is mapped to :math:`-i(id \otimes H(t) - H(t)^T \otimes id)`,
        and a dissipation term
        :math:`X \mapsto LXL^\dagger - \frac{1}{2}\{L^\dagger L,X\}` is
        given as
        :math:`\overline{L} \otimes L - \frac{1}{2}(id \otimes L^\daggerL + \overline{L^\dagger L} \otimes id)`.

        This function turns every operator in the :class:`HamiltonianModel`
        into the vectorized version of :math:`-i[H_j, \cdot]`, every
        operator in `self.noise_operators` into the vectorized dissipator
        above, and concatenates these lists of operators, as well as the
        signals corresponding to both, to form a new :class:`OperatorModel`.

        Returns:
            OperatorModel: corresponding to vectorized Lindblad equation.
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
