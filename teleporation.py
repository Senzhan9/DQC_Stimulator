from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qiskit.quantum_info import Kraus
from qiskit_aer.noise import depolarizing_error

from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

a1, b1 = 1/2**0.5, 1/2**0.5   # q0 = (|0> + |1>)/√2
a2, b2 = 1, 0                 # q1 = |0>

# 定义退极化噪声（单比特，错误率 5%）
error = depolarizing_error(0.5, 1)
# 转换成 Qiskit 可以 append 的 instruction
noise_channel = error.to_instruction()

# 建立一个 4 qubits + 4 clbits 的电路
qc = QuantumCircuit(4, 4)
# 初始化态
qc.initialize([a1, b1], 0)  # q0 初态
qc.initialize([a2, b2], 3)  # q1 初态

# Step 1: 在 q2 和 q1 上生成 Bell 态 (|00> + |11>)
qc.h(1)        # 对 q2 施加 H
qc.cx(1, 2)    # q2 -> q1
qc.append(noise_channel, [2])       # 在 q1 上插入一次噪声

# Step 2: CNOT
qc.cx(2, 3)    # q3 -> q2
qc.cx(0, 1)    # q1 -> q0

# Step 3: 对 q2 施加 H
qc.h(2)

# Step 4: 测量 q1 -> c1, q2 -> c2
qc.measure(1, 1)
qc.measure(2, 2)

# Step 5: 条件操作
with qc.if_test((1, 1)):  # 如果 c1==1
    qc.x(3)

with qc.if_test((2, 1)):  # 如果 c2==1
    qc.z(0)

# Step 6: 最后测量 q0, q3
qc.measure(0, 0)  # q0 -> c2
qc.measure(3, 3)  # q3 -> c3

# 使用 Aer 模拟
sim = AerSimulator()
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=1000).result()
counts = result.get_counts()


# ----- extract ---------
def extract_q0_q3(counts):
    new_counts = {}
    for k, v in counts.items():
        # k = 'c3c2c1c0'
        key = k[0] + k[3]
        if key in new_counts:
            new_counts[key] += v
        else:
            new_counts[key] = v
    return new_counts

q0_q3_counts = extract_q0_q3(counts)
# -----------------------

print("Results：", q0_q3_counts)

# 
qc.draw("mpl")
plot_histogram(q0_q3_counts)
plt.show()