# =========================
# Python path setup
# =========================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import collections
from typing import List, Tuple
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import NoiseModel

# =========================
# Fake backends (heterogeneous QPUs)
# =========================
from qiskit_ibm_runtime.fake_provider import (
    FakeVigoV2,
    FakeLagosV2,
    FakeCasablancaV2,
    FakeYorktownV2,
    FakeManilaV2,
    FakeNairobiV2,
    FakeMumbaiV2,
    FakeKolkataV2,
    FakeGuadalupeV2,
    FakeAlmadenV2,
    FakeAthensV2,
    FakeCambridgeV2
)

# =========================
# Distributed Quantum Computing Simulator
# =========================
from dqc_simulator import DQCCircuit, DQCQPU, QPUManager
from dqc_simulator.backend import IonQ
from dqc_simulator import extract_feature_vector, extract_feature_groups

# ============================================================
# Fidelity metric: Hellinger fidelity for GHZ state
# ============================================================
def score(exp_probs: dict) -> float:
    """
    Compute Hellinger fidelity for an n-qubit GHZ state.

    Args:
        exp_probs: Experimental probability distribution
                   e.g., {'000...0': p0, '111...1': p1}

    Returns:
        fidelity in [0, 1]
    """
    n = len(next(iter(exp_probs)))

    # Ideal GHZ distribution
    ideal_probs = {
        '0' * n: 0.5,
        '1' * n: 0.5
    }

    # Hellinger fidelity
    fidelity = sum(
        np.sqrt(ideal_probs.get(k, 0)) * np.sqrt(p)
        for k, p in exp_probs.items()
    )
    return fidelity


# ============================================================
# Step 1: Construct global GHZ circuit (12 qubits)
# ============================================================
numbits = 12
qc0 = QuantumCircuit(numbits, numbits)

# GHZ state preparation
qc0.h(0)
for i in range(numbits - 1):
    qc0.cx(i, i + 1)

# Measurement on all qubits
for i in range(numbits):
    qc0.measure(i, i)

for instr in qc0.data:
    print(instr)

# ============================================================
# Step 2: Transform into Distributed Quantum Circuit
# ============================================================
qc = DQCCircuit(qc0)

# Partition the circuit across 4 QPUs
Partition = [3, 3, 3, 3]


# ============================================================
# Step 3: Configure heterogeneous QPU group
# ============================================================
QPUGROUP = QPUManager()

# Add 4 QPUs (all FakeVigoV2 here, but framework supports heterogeneity)
QPUGROUP.add_qpu(DQCQPU(0, "FakeVigoV2"))
QPUGROUP.add_qpu(DQCQPU(1, "FakeVigoV2"))
QPUGROUP.add_qpu(DQCQPU(2, "FakeVigoV2"))
QPUGROUP.add_qpu(DQCQPU(3, "FakeVigoV2"))

# Define inter-QPU network topology
dis = 5  # communication distance / cost
QPUGROUP.add_coonnection(0, 1, distance=dis)
QPUGROUP.add_coonnection(1, 2, distance=dis)
QPUGROUP.add_coonnection(2, 3, distance=dis)


# ============================================================
# Step 4: Distributed execution with communication noise
# ============================================================
result_qc = qc.Execution(
    Partition,
    QPUGROUP,
    comm_noise=True
)

# Obtain combined noise model (local + communication)
noise_model = qc.get_noise_model()

# Backend simulator
sim = AerSimulator(noise_model=noise_model)

# Transpile for backend
compiled = transpile(result_qc, sim)

feature_vector = extract_feature_vector(compiled)
feature_groups = extract_feature_groups(compiled)

print("Feature vector:", feature_vector)
print("Communication count (kraus):", feature_vector.get("comm_count"))
print(
    "Communication noise mean/max/min/std:",
    feature_groups["group_c_noise_strength"]["comm_noise_mean"],
    feature_groups["group_c_noise_strength"]["comm_noise_max"],
    feature_groups["group_c_noise_strength"]["comm_noise_min"],
    feature_groups["group_c_noise_strength"]["comm_noise_std"],
)
# for instr in compiled.data:
#     print(instr)

# compiled.draw("mpl", scale=0.7, fold=100)
# plt.show()

# # Run simulation
# job = sim.run(compiled, shots=10_000)
# result = job.result()


# # ============================================================
# # Step 5: Post-processing measurement results
# # ============================================================
# counts = result.get_counts()

# # Only keep the first 12 bits (global logical qubits)
# counts_res = {}
# for bitstring, cnt in counts.items():
#     bits = bitstring[:12]
#     counts_res[bits] = counts_res.get(bits, 0) + cnt


# # ============================================================
# # Step 6: Visualization and PDF export
# # ============================================================

# # --- 6.1 Histogram ---
# fig1 = plot_histogram(counts_res)
# fig1.savefig("GHZ_12qubit_DQC_histogram.pdf")


# # --- 6.2 Circuit diagram ---
# fig2 = result_qc.draw("mpl", scale=0.7, fold=100)
# fig2.savefig("GHZ_12qubit_DQC_circuit.pdf")

# plt.show()
