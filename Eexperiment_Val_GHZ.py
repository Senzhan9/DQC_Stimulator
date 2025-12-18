import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import collections
from typing import List, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import collections
from dqc_simulator import DQCCircuit, DQCQPU, QPUManager
from dqc_simulator.backend import IonQ
import matplotlib.pyplot as plt
from qiskit_ibm_runtime.fake_provider import (
    FakeVigoV2,         # 5
    FakeLagosV2,        # 7
    FakeCasablancaV2,
    FakeYorktownV2,
    FakeManilaV2,
    FakeNairobiV2,
    FakeMumbaiV2,
    FakeKolkataV2,
    FakeGuadalupeV2,
    FakeAlmadenV2,
    FakeAthensV2,       # 5
    FakeCambridgeV2                
)
from qiskit_aer.noise import NoiseModel
from qiskit.visualization import plot_histogram

def score(exp_probs: dict) -> float:
    """
    Compute Hellinger fidelity for an n-qubit GHZ state.

    Args:
        exp_probs: dict, experimental probability distribution, e.g.,
                   {'000': 0.49, '111': 0.51, ...}

    Returns:
        fidelity: float in [0,1]
    """
    n = len(next(iter(exp_probs)))  # 取任意比特串长度作为 qubit 数
    ideal_probs = {
        '0'*n: 0.5,
        '1'*n: 0.5
    }

    # Hellinger fidelity
    fidelity = sum(np.sqrt(ideal_probs.get(k, 0)) * np.sqrt(p) for k, p in exp_probs.items())
    return fidelity

numbits = 12
qc0 = QuantumCircuit(numbits, numbits)
qc0.h(0)
qc0.cx(0,1)
qc0.cx(1,2)
qc0.cx(2,3)
qc0.cx(3,4)
qc0.cx(4,5)
qc0.cx(5,6)
qc0.cx(6,7)
qc0.cx(7,8)
qc0.cx(8,9)
qc0.cx(9,10)
qc0.cx(10,11)

for i in range(numbits):
    qc0.measure(i,i)

qc = DQCCircuit(qc0)
Partition = [3,3,3,3]
QPUGROUP = QPUManager()
QPUGROUP.add_qpu(DQCQPU(0, "FakeVigoV2"))
QPUGROUP.add_qpu(DQCQPU(1, "FakeVigoV2"))
QPUGROUP.add_qpu(DQCQPU(2, "FakeVigoV2"))
QPUGROUP.add_qpu(DQCQPU(3, "FakeVigoV2"))

dis = 5
QPUGROUP.add_coonnection(0,1,distance=dis)
# QPUGROUP.add_coonnection(0,2,distance=dis)
# QPUGROUP.add_coonnection(0,3,distance=dis)+
QPUGROUP.add_coonnection(1,2,distance=dis)
# QPUGROUP.add_coonnection(1,3,distance=dis)
QPUGROUP.add_coonnection(2,3,distance=dis)

result_qc = qc.Execution(Partition, QPUGROUP, comm_noise=True)
result_qc.draw("mpl", scale=0.7, fold=100)
NoiseModel = qc.get_noise_model()
sim = AerSimulator(noise_model=NoiseModel)
compiled = transpile(result_qc, sim)
job = sim.run(compiled, shots=1000) 
result = job.result()

counts = result.get_counts()

counts_1 = {}
for bitstring, cnt in counts.items():
    bits = bitstring[:12]
    counts_1[bits] = counts_1.get(bits, 0) + cnt

plot_histogram(counts_1)
plt.show()

