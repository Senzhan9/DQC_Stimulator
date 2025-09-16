from DQCCircuit import DQCCircuit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Step 1: 创建 2-qubit 电路
qc = DQCCircuit(4)
qc.Generate_COMM_Qubit([1, 2])

# Step 2: 获取退极化噪声 Instruction
noise_instr = DQCCircuit.get_noise_channel("depolarizing", p=0.5)

# Step 3: 生成 Bell 态，并对 target qubit 加噪声
qc.Generate_Bell(qc=1, qt=2, noise_channel=noise_instr)

#
qc.TeleGate(0,1,3,2)

# Step 4: 测量
qc.measure_all()

# Step 5: Aer模拟器运行
sim = AerSimulator()
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=1000).result()

# Step 6: 获取测量结果并绘制直方图
counts = result.get_counts()
print("Measurement results:", counts)
qc.draw("mpl")
plot_histogram(counts)
plt.show()