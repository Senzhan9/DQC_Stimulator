# Quantum Circuit Functions Documentation

## Table of Contents
1. [Generate_COMM_Qubit](#generate_comm_qubit)
2. [Generate_Bell](#generate_bell)
3. [get_noise_channel](#get_noise_channel)
4. [TeleGate](#telegate)

---

## Generate_COMM_Qubit

**Description:**  
Mark specified qubits as communication qubits.

**Arguments:**  

| Parameter      | Type       | Description                                      |
|----------------|-----------|--------------------------------------------------|
| `comm_qubits`  | list[int] | A list of qubit indices to be marked as COMM qubits. |

**Raises:**  
- `ValueError`: If any index in `comm_qubits` is out of the valid range `[0, self.num_qubits - 1]`.

**Notes:**  
- Updates the `self.qubit_type` array, setting the type of each specified qubit to 1 (COMM qubit).

---

## Generate_Bell

**Description:**  
Generate a Bell state (|00> + |11>) between two communication qubits.

**Arguments:**  

| Parameter       | Type       | Description                                                                 |
|-----------------|-----------|-----------------------------------------------------------------------------|
| `qc`            | int       | Index of the control qubit.                                                 |
| `qt`            | int       | Index of the target qubit.                                                  |
| `noise_channel` | optional  | A Qiskit noise channel applied to the target qubit. Default is `None`.      |
| `error_rate`    | float     | Probability of transmission error, must be in [0,1]. Default is 0.0.        |

**Raises:**  
- `ValueError`: If either `qc` or `qt` is not a communication qubit.  
- `ValueError`: If `error_rate` is outside the range [0,1].

**Returns:**  
- `QuantumCircuit`: The modified circuit with the generated Bell state.

**Notes:**  
- Applies a Hadamard gate to the control qubit, followed by a CNOT gate.  
- If `noise_channel` is provided, it simulates transmission noise.  
- `error_rate` can represent transmission errors or retransmission.

---

## get_noise_channel

**Description:**  
Return a Qiskit noise instruction corresponding to the specified noise type.

**Arguments:**  

| Parameter      | Type       | Description                                                                 |
|----------------|-----------|-----------------------------------------------------------------------------|
| `noise_type`   | str       | Type of noise. Options: 'depolarizing', 'bitflip', 'Yflip', 'phaseflip', 'amplitude_damping', 'generalized_amplitude_damping', 'phase_damping', 'kraus'. |
| `**kwargs`     | dict      | Additional parameters depending on noise type: `p`, `gamma`, `K0`, `K1`.   |

**Raises:**  
- `ValueError`: If `p` is provided but not in [0,1].  
- `ValueError`: If `noise_type` is invalid.

**Returns:**  
- `Instruction`: Qiskit Instruction representing the noise channel.

**Notes:**  
- Probabilistic errors use `pauli_error` or `depolarizing_error`.  
- Damping errors use Qiskit damping models.  
- For 'kraus', custom Kraus operators can be passed via kwargs.

---

## TeleGate

**Description:**  
Perform a teleportation-based CNOT gate (TeleGate) using communication qubits.

**Arguments:**  

| Parameter              | Type | Description                                                              |
|------------------------|------|--------------------------------------------------------------------------|
| `qubit_control`        | int  | Index of the control qubit in the main circuit.                          |
| `comm_qubit_control`   | int  | Communication qubit corresponding to the control qubit.                 |
| `qubit_target`         | int  | Index of the target qubit in the main circuit.                           |
| `comm_qubit_target`    | int  | Communication qubit corresponding to the target qubit.                  |

**Raises:**  
- `ValueError`: If any qubit index is out of [0, self.num_qubits-1].  
- `ValueError`: If either communication qubit is not a communication qubit.

**Returns:**  
- `QuantumCircuit`: Circuit with teleportation-based CNOT applied.

**Notes:**  
1. Performs CNOT gates between main qubits and their communication qubits.  
2. Applies Hadamard gate to target communication qubit.  
3. Measures communication qubits and stores in classical register.  
4. Applies conditional X/Z gates based on measurements.  
5. Assumes communication qubits were set with `Generate_COMM_Qubit`.

---
