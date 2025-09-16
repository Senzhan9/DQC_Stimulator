from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qiskit_aer.noise import (
    depolarizing_error, pauli_error,
    amplitude_damping_error, phase_amplitude_damping_error,
    phase_damping_error
)
from qiskit.quantum_info import Kraus

import numpy as np

class DQCCircuit(QuantumCircuit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Inherit initialization from the parent class

    # qc:Control Qubit | qt: Target Qubit
    def Generate_Bell(self, qc: int, qt: int, noise_channel=None, error_rate: float = 0.0):
        # Generating Bell states (|00>+|11>) in the quantum circuit
        self.h(qc)
        self.cx(qc, qt)

        # noise_channel
        if noise_channel:
            noise_channel.label = "Transmission noise"
            self.append(noise_channel, [qt])
        # error_rate 传送错误，重传

        return self
    
    def get_noise_channel(noise_type: str, **kwargs):
    # 根据 noise_type 返回对应的噪声 Instruction
    # 参数:
    #     noise_type: str, 支持
    #         'depolarizing', 'bitflip', 'phaseflip', 'amplitude_damping',
    #         'generalized_amplitude_damping', 'phase_damping', 'kraus'
    #     kwargs: dict, 每种噪声的额外参数
    # 返回:
    #     Qiskit Instruction
    
        # Check p, ensure it valid. 
        p = kwargs.get("p", None)  # 如果没传 p，就返回 None
        if p is not None and not (0 <= p <= 1):
            raise ValueError(f"Invalid probability p={p}, must be in [0, 1].")

        if noise_type == "depolarizing":
            p = kwargs.get("p", 0.1)
            error = depolarizing_error(p, 1)
        elif noise_type == "bitflip":
            p = kwargs.get("p", 0.1)
            error = pauli_error([('X', p), ('I', 1-p)])
        elif noise_type == "Yflip":
            p = kwargs.get("p", 0.1)
            error = pauli_error([('Y', p), ('I', 1-p)])
        elif noise_type == "phaseflip":
            p = kwargs.get("p", 0.1)
            error = pauli_error([('Z', p), ('I', 1-p)])
        elif noise_type == "amplitude_damping":
            gamma = kwargs.get("gamma", 0.1)
            error = amplitude_damping_error(gamma)
        elif noise_type == "generalized_amplitude_damping":
            gamma = kwargs.get("gamma", 0.1)
            p = kwargs.get("p", 0.3)
            error = phase_amplitude_damping_error(gamma, p)
        elif noise_type == "phase_damping":
            gamma = kwargs.get("gamma", 0.1)
            error = phase_damping_error(gamma)
        elif noise_type == "kraus":
            K0 = kwargs.get("K0", np.sqrt(0.9) * np.eye(2))
            K1 = kwargs.get("K1", np.sqrt(0.1) * np.array([[1, 0], [0, -1]]))
            error = Kraus([K0, K1])
        else:
            raise ValueError(f"Noise Type Error: {noise_type}")

        return error.to_instruction()

