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

# pylint: disable=invalid-name
"""
Qiskit Aer simulator backend utils
"""
import os
from math import log2
from qiskit.util import local_hardware_info
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import assemble

# Available system memory
SYSTEM_MEMORY_GB = local_hardware_info()['memory']

# Max number of qubits for complex double statevector
# given available system memory
MAX_QUBITS_STATEVECTOR = int(log2(SYSTEM_MEMORY_GB * (1024**3) / 16))

# Location where we put external libraries that will be
# loaded at runtime by the simulator extension
LIBRARY_DIR = os.path.dirname(__file__)


def cpp_execute(controller, qobj):
    """Execute qobj_dict on C++ controller wrapper"""
    # Location where we put external libraries that will be
    # loaded at runtime by the simulator extension
    qobj['config']['library_dir'] = LIBRARY_DIR
    return controller(qobj)


def available_methods(controller, methods):
    """Check available simulation methods by running a dummy circuit."""
    # Test methods are available using the controller
    dummy_circ = QuantumCircuit(1)
    # NOTE: This hasattr check is to remove derecation warning
    # for changing `iden` to `i`, while maintaining backwards
    # compatibility. It should be removed once `i` has made it
    # into Terra stable.
    if hasattr(dummy_circ, 'i'):
        dummy_circ.i(0)
    else:
        dummy_circ.iden(0)

    valid_methods = []
    for method in methods:
        qobj = assemble(dummy_circ,
                        optimization_level=0,
                        method=method).to_dict()
        result = cpp_execute(controller, qobj)
        if result.get('success', False):
            valid_methods.append(method)
    return valid_methods


def backend_gates(names):
    """Return backend gates list.

    Args:
        names (list): basis gate names.

    Returns:
        list: list of basis gate descriptions.
    """
    if isinstance(names, str):
        names = [names]
    gates = []
    for name in names:
        gate_dict = getattr(BackendGates, name, None)
        if gate_dict:
            gates.append(gate_dict)
    return gates


class BackendGates:
    """Struct of basis gate dictionaries for backend configurations."""

    u1 = {
        'name': 'u1',
        'parameters': ['lam'],
        'conditional': True,
        'description': 'Single-qubit gate [[1, 0], [0, exp(1j*lam)]]',
        'qasm_def': 'gate u1(lam) q { U(0,0,lam) q; }'
    }

    u2 = {
        'name': 'u2',
        'parameters': ['phi', 'lam'],
        'conditional': True,
        'description':
        'Single-qubit gate [[1, -exp(1j*lam)], [exp(1j*phi), exp(1j*(phi+lam))]]/sqrt(2)',
        'qasm_def': 'gate u2(phi,lam) q { U(pi/2,phi,lam) q; }'
    }

    u3 = {
        'name': 'u3',
        'parameters': ['theta', 'phi', 'lam'],
        'conditional': True,
        'description': 'Single-qubit gate with three rotation angles',
        'qasm_def': 'gate u3(theta,phi,lam) q { U(theta,phi,lam) q; }'
    }

    cx = {
        'name': 'cx',
        'parameters': [],
        'conditional': True,
        'description': 'Two-qubit Controlled-NOT gate',
        'qasm_def': 'gate cx c,t { CX c,t; }'
    }

    cz = {
        'name': 'cz',
        'parameters': [],
        'conditional': True,
        'description': 'Two-qubit Controlled-Z gate',
        'qasm_def': 'gate cz a,b { h b; cx a,b; h b; }'
    }

    iden = {
        'name': 'id',
        'parameters': [],
        'conditional': True,
        'description': 'Single-qubit identity gate',
        'qasm_def': 'gate id a { U(0,0,0) a; }'
    }

    x = {
        'name': 'x',
        'parameters': [],
        'conditional': True,
        'description': 'Single-qubit Pauli-X gate',
        'qasm_def': 'gate x a { U(pi,0,pi) a; }'
    }

    y = {
        'name': 'y',
        'parameters': [],
        'conditional': True,
        'description': 'Single-qubit Pauli-Y gate',
        'qasm_def': 'TODO'
    }

    z = {
        'name': 'z',
        'parameters': [],
        'conditional': True,
        'description': 'Single-qubit Pauli-Z gate',
        'qasm_def': 'TODO'
    }

    h = {
        'name': 'h',
        'parameters': [],
        'conditional': True,
        'description': 'Single-qubit Hadamard gate',
        'qasm_def': 'TODO'
    }

    s = {
        'name': 's',
        'parameters': [],
        'conditional': True,
        'description': 'Single-qubit phase gate',
        'qasm_def': 'TODO'
    }

    sdg = {
        'name': 'sdg',
        'parameters': [],
        'conditional': True,
        'description': 'Single-qubit adjoint phase gate',
        'qasm_def': 'TODO'
    }

    t = {
        'name': 't',
        'parameters': [],
        'conditional': True,
        'description': 'Single-qubit T gate',
        'qasm_def': 'TODO'
    }

    tdg = {
        'name': 'tdg',
        'parameters': [],
        'conditional': True,
        'description': 'Single-qubit adjoint T gate',
        'qasm_def': 'TODO'
    }

    swap = {
        'name': 'swap',
        'parameters': [],
        'conditional': True,
        'description': 'Two-qubit SWAP gate',
        'qasm_def': 'TODO'
    }

    ccx = {
        'name': 'ccx',
        'parameters': [],
        'conditional': True,
        'description': 'Three-qubit Toffoli gate',
        'qasm_def': 'TODO'
    }

    cswap = {
        'name': 'cswap',
        'parameters': [],
        'conditional': True,
        'description': 'Three-qubit Fredkin (controlled-SWAP) gate',
        'qasm_def': 'TODO'
    }

    unitary = {
        'name': 'unitary',
        'parameters': ['matrix'],
        'conditional': True,
        'description': 'N-qubit arbitrary unitary gate. '
                       'The parameter is the N-qubit matrix to apply.',
        'qasm_def': 'unitary(matrix) q1, q2,...'
    }

    initialize = {
        'name': 'initialize',
        'parameters': ['vector'],
        'conditional': False,
        'description': 'N-qubit state initialize. Resets qubits '
                       'then sets statevector to the parameter vector.',
        'qasm_def': 'initialize(vector) q1, q2,...'
    }

    cu1 = {
        'name': 'cu1',
        'parameters': ['lam'],
        'conditional': True,
        'description': 'Two-qubit Controlled-u1 gate',
        'qasm_def': 'TODO'
    }

    cu2 = {
        'name': 'cu2',
        'parameters': ['phi', 'lam'],
        'conditional': True,
        'description': 'Two-qubit Controlled-u2 gate',
        'qasm_def': 'TODO'
    }

    cu3 = {
        'name': 'cu3',
        'parameters': ['theta', 'phi', 'lam'],
        'conditional': True,
        'description': 'Two-qubit Controlled-u3 gate',
        'qasm_def': 'TODO'
    }

    mcx = {
        'name': 'mcx',
        'parameters': [],
        'conditional': True,
        'description': 'N-qubit multi-controlled-X gate',
        'qasm_def': 'TODO'
    }

    mcy = {
        'name': 'mcy',
        'parameters': [],
        'conditional': True,
        'description': 'N-qubit multi-controlled-Y gate',
        'qasm_def': 'TODO'
    }

    mcz = {
        'name': 'mcz',
        'parameters': [],
        'conditional': True,
        'description': 'N-qubit multi-controlled-Z gate',
        'qasm_def': 'TODO'
    }

    mcu1 = {
        'name': 'mcu1',
        'parameters': ['lam'],
        'conditional': True,
        'description': 'N-qubit multi-controlled-u1 gate',
        'qasm_def': 'TODO'
    }

    mcu2 = {
        'name': 'mcu2',
        'parameters': ['phi', 'lam'],
        'conditional': True,
        'description': 'N-qubit multi-controlled-u2 gate',
        'qasm_def': 'TODO'
    }

    mcu3 = {
        'name': 'mcu3',
        'parameters': ['theta', 'phi', 'lam'],
        'conditional': True,
        'description': 'N-qubit multi-controlled-u3 gate',
        'qasm_def': 'TODO'
    }

    mcswap = {
        'name': 'mcswap',
        'parameters': [],
        'conditional': True,
        'description': 'N-qubit multi-controlled-SWAP gate',
        'qasm_def': 'TODO'
    }

    multiplexer = {
        'name': 'multiplexer',
        'parameters': ['mat1', 'mat2', '...'],
        'conditional': True,
        'description': 'N-qubit multi-plexer gate. '
                       'The input parameters are the gates for each value.',
        'qasm_def': 'TODO'
    }

    kraus = {
        'name': 'kraus',
        'parameters': ['mat1', 'mat2', '...'],
        'conditional': True,
        'description': 'N-qubit Kraus error instruction. '
                       'The input parameters are the Kraus matrices.',
        'qasm_def': 'TODO'
    }

    roerror = {
        'name': 'roerror',
        'parameters': ['matrix'],
        'conditional': False,
        'description': 'N-bit classical readout error instruction. '
                       'The input parameter is the readout error probability matrix.',
        'qasm_def': 'TODO'
    }
