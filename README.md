# Distributed Quantum Circuit 

This repository demonstrates the creation, partitioning, execution, and visualization of a distributed quantum circuit using **DQCCircuit** and QPU simulation. It also shows how to save measurement results and circuit diagrams as images.

---

## Example Environment

- **Python**: 3.13.X
- **Qiskit**
- **Qiskit Aer**
- **Qiskit IBM Runtime**
- **Matplotlib**
- **pylatexenc**

Install dependencies with pip:

```bash
# Make sure you are using Python 3.13.5
pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib pylatexenc
```

## Quick Start Guide

### 1. Define Circuit Size
Set the number of qubits for your quantum circuit:
```python
numbits = 12  # Total logical qubits
```

### 2. Create Global Quantum Circuit
Build your quantum algorithm (e.g., GHZ state preparation):
```python
qc0 = QuantumCircuit(numbits, numbits)
qc0.h(0)
for i in range(numbits - 1):
    qc0.cx(i, i + 1)
# Add measurements
for i in range(numbits):
    qc0.measure(i, i)
```

### 3. Convert to Distributed Circuit
Transform the standard circuit into a distributed format:
```python
qc = DQCCircuit(qc0)
```

### 4. Define Circuit Partition
Specify how to distribute qubits across QPUs:
```python
# Option 1: Equal distribution by count
Partition = [3, 3, 3, 3]  # 4 QPUs, 3 qubits each

# Option 2: Explicit qubit assignment
Partition = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
```

### 5. Configure QPU Manager
Set up heterogeneous QPU group with network topology:
```python
QPUGROUP = QPUManager()

# Add QPUs with different backends
QPUGROUP.add_qpu(DQCQPU(0, "FakeVigoV2"))
QPUGROUP.add_qpu(DQCQPU(1, "FakeLagosV2"))
QPUGROUP.add_qpu(DQCQPU(2, "FakeAthensV2"))
QPUGROUP.add_qpu(DQCQPU(3, "FakeManilaV2"))

# Define inter-QPU connections
dis = 5  # Communication distance/cost
QPUGROUP.add_coonnection(0, 1, distance=dis)
QPUGROUP.add_coonnection(1, 2, distance=dis)
QPUGROUP.add_coonnection(2, 3, distance=dis)
```

### 6. Execute Distributed Circuit
Run with communication noise modeling:
```python
result_qc = qc.Execution(
    Partition,
    QPUGROUP,
    comm_noise=True  # Enable teleportation noise
)
```

### 7. Simulate and Analyze Results
Execute on AerSimulator with combined noise model:
```python
# Get combined noise model (local QPU + communication)
noise_model = qc.get_noise_model()
sim = AerSimulator(noise_model=noise_model)

# Transpile and run
compiled = transpile(result_qc, sim)
job = sim.run(compiled, shots=10_000)
result = job.result()
```

### 8. Visualize and Export
Generate publication-ready figures:
```python
# Measurement histogram
fig1 = plot_histogram(counts_res)
fig1.savefig("GHZ_12qubit_DQC_histogram.pdf")

# Circuit diagram
fig2 = result_qc.draw("mpl", scale=0.7, fold=100)
fig2.savefig("GHZ_12qubit_DQC_circuit.pdf")
```
