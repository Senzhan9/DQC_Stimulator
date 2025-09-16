from qiskit import QuantumCircuit, ClassicalRegister, transpile
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
        # 每个 qubit 的类型数组，0=计算比特，1=通信比特，初始化为计算比特
        self.qubit_type = [0] * self.num_qubits

    def Generate_COMM_Qubit(self, comm_qubits: list[int]):
        # 传入List数组，转化对应位置为COMM Qubit
        for idx in comm_qubits:
            if 0 <= idx < self.num_qubits:
                self.qubit_type[idx] = 1
            else:
                raise ValueError(f"Invalid qubit index {idx}, must be in [0, {self.num_qubits-1}]")

    # qc:Control Qubit | qt: Target Qubit
    def Generate_Bell(self, qc: int, qt: int, noise_channel=None, error_rate: float = 0.0):
        # 检查 qc 和 qt 是否为通信比特
        if self.qubit_type[qc] != 1 or self.qubit_type[qt] != 1:
            raise ValueError(f"Both qubits must be communication qubits.")
        
        # Generating Bell states (|00>+|11>) in the quantum circuit
        self.h(qc)
        self.cx(qc, qt)

        # noise_channel
        if noise_channel:
            noise_channel.label = "Transmission noise"
            self.append(noise_channel, [qt])
        # error_rate 传送错误，重传
        if not (0 <= error_rate <= 1):
            raise ValueError(f"Invalid probability error_rate={error_rate}, must be in [0, 1].")
        
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
    
    # 
    def TeleGate(self, qubit_control: int, comm_qubit_control: int, qubit_target: int, comm_qubit_target: int):
 
        # 检查索引范围
        all_qubits = [qubit_control, qubit_target, comm_qubit_control, comm_qubit_target]
        if not all(0 <= q < self.num_qubits for q in all_qubits):
            raise ValueError("All qubit indices must be within [0, self.num_qubits-1].")

        # 检查 comm_qubits 是否都是通信比特
        if not all(self.qubit_type[q] == 1 for q in [comm_qubit_control, comm_qubit_target]):
            raise ValueError("Both comm_qubit_control and comm_qubit_target must be communication qubits.")
 
        # Step 2: CNOT
        self.cx(qubit_control, comm_qubit_control)  
        self.cx(comm_qubit_target, qubit_target)   

        # Step 3: 对 comm_qubit_target 施加 H
        self.h(comm_qubit_target)

        # Step 4: 测量
        classical_Channel = ClassicalRegister(2, 'tele')
        self.add_register(classical_Channel)
        self.measure(comm_qubit_control, classical_Channel[0])
        self.measure(comm_qubit_target, classical_Channel[1])

        # Step 5: 条件操作
        with self.if_test((classical_Channel[0], 1)):  # x-flip
            self.x(qubit_target)

        with self.if_test((classical_Channel[1], 1)):  # z-flip
            self.z(qubit_control)

        return self
