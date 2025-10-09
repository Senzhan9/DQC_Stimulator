# Distributed Quantum Circuit 

This repository demonstrates the creation, partitioning, execution, and visualization of a distributed quantum circuit using **DQCCircuit** and QPU simulation. It also shows how to save measurement results and circuit diagrams as images.

---

## Example Environment

- **Python**: 3.13.5
- **Qiskit**
- **Qiskit Aer**
- **Qiskit IBM Runtime**
- **Matplotlib**

Install dependencies with pip:

```bash
# Make sure you are using Python 3.13.5
pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib
```

## Quick Start

### 1. Set the number of qubits
Modify `num_qubits` to control the size of the simulated circuit.

### 2. Create the quantum circuit
Build gates and measurement operations.

### 3. Partition the circuit
Use the `Partition` variable to split the circuit. Supported forms:

- Number list: `[2,2,2]`
- Explicit qubit indices: `[[1,3],[0,2],[4,5]]`

### 4. Create QPUs and assign backends
Each QPU has an `index`, which will determine which sub-circuit it compiles.  
**Note:** Each backend supports a limited number of qubits for transpilation.

```python
qpu1 = DQCQPU(0, "FakeVigoV2")
qpu2 = DQCQPU(1, "FakeAthensV2")
qpu3 = DQCQPU(2, "FakeLagosV2")
```

### 5. Execute the circuit 
```python
result_qc = qc.Execution(Partition, [qpu1, qpu2, qpu3])
```

### 6. Simulate and save output 
Run the circuit on AerSimulator and save the measurement histogram and circuit diagrams as images.

---

##  使用指南

### 1. 设置量子比特数量
修改 `num_qubits` 来控制模拟电路的规模。

### 2. 创建量子电路
构建量子门和测量操作。

### 3. 划分电路
使用 `Partition` 变量对电路进行划分。支持的形式：

- 按数量划分: `[2,2,2]`
- 按指定比特索引划分: `[[1,3],[0,2],[4,5]]`

### 4. 创建 QPU 并指定后端
每个 QPU 有一个 `index`，它将决定对应的子电路编译顺序。  
**注意：** 每个后端支持的可 transpile 的比特数量有限。

```python
qpu1 = DQCQPU(0, "FakeVigoV2")
qpu2 = DQCQPU(1, "FakeAthensV2")
qpu3 = DQCQPU(2, "FakeLagosV2")
```

### 5. 执行电路
```python
result_qc = qc.Execution(Partition, [qpu1, qpu2, qpu3])
```

### 6. 模拟并保存输出
在 AerSimulator 上运行电路，并将测量结果直方图和电路图保存为图片。
