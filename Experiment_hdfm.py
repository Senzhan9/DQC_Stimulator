"""
量子哈密顿量模拟 - 1D横场伊辛模型(TFIM)
使用Qiskit实现，支持分布式量子计算(DQC)
"""
# =========================
# Python path setup
# =========================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import time
import collections
from typing import List, Tuple
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit.converters import circuit_to_dag
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

class HamiltonianSimulationQiskit:
    """
    1D横场伊辛模型(TFIM)哈密顿量模拟的Qiskit实现
    
    Parameters:
    -----------
    num_qubits : int
        量子比特数量
    time_step : int
        时间步长
    total_time : int
        总演化时间
    """

    def __init__(self, num_qubits: int, time_step: int, total_time: int) -> None:
        self.num_qubits = num_qubits
        self.time_step = time_step
        self.total_time = total_time

    def circuit(self) -> QuantumCircuit:
        """
        生成量子电路
        
        Returns:
        --------
        QuantumCircuit : 构建的量子电路
        """
        # 物理常数
        hbar = 0.658212  # eV*fs (约化普朗克常数)
        jz = hbar * np.pi / 4  # eV, 耦合系数
        freq = 0.0048  # 1/fs, MoSe2声子频率

        # 声子参数
        w_ph = 2 * np.pi * freq
        e_ph = 3 * np.pi * hbar / (8 * np.cos(np.pi * freq))

        # 初始化量子电路
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # 构建时间演化电路
        num_steps = int(self.total_time / self.time_step)
        for step in range(num_steps):
            t = (step + 0.5) * self.time_step

            # 单量子比特项 (横场项)
            psi = -2.0 * e_ph * np.cos(w_ph * t) * self.time_step / hbar
            for q in range(self.num_qubits):
                qc.h(q)
                qc.rz(psi, q)
                qc.h(q)

            # 双量子比特耦合项 (伊辛相互作用)
            psi2 = -2.0 * jz * self.time_step / hbar
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(psi2, i + 1)
                qc.cx(i, i + 1)

        # 添加测量
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def _get_ideal_counts(self, circuit: QuantumCircuit) -> collections.Counter:
        """
        计算理想(无噪声)概率分布
        
        Parameters:
        -----------
        circuit : QuantumCircuit
            量子电路
            
        Returns:
        --------
        Counter : 理想概率分布
        """
        qc_no_measure = circuit.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_measure)
        probs = sv.probabilities_dict()
        return collections.Counter(probs)

    def _average_magnetization(self, result: dict, shots: int) -> float:
        """
        计算平均磁化强度 <Z>
        
        Parameters:
        -----------
        result : dict
            测量结果计数
        shots : int
            总测量次数
            
        Returns:
        --------
        float : 平均磁化强度
        """
        mag = 0
        for spin_str, count in result.items():
            bitstring = spin_str[::-1]
            # 将比特串转换为自旋: 0→+1, 1→-1
            spin_int = [1 - 2 * int(s) for s in bitstring]
            mag += (sum(spin_int) / len(spin_int)) * count
        average_mag = mag / shots
        return average_mag

    def score(self, counts: collections.Counter) -> float:
        """
        基于磁化强度计算保真度评分
        
        Parameters:
        -----------
        counts : Counter
            实验测量结果
            
        Returns:
        --------
        float : [0,1]范围的评分，1表示完美匹配
        """
        ideal_counts = self._get_ideal_counts(self.circuit())
        total_shots = sum(counts.values())

        mag_ideal = self._average_magnetization(ideal_counts, 1)
        mag_experimental = self._average_magnetization(counts, total_shots)

        return 1 - abs(mag_ideal - mag_experimental) / 2

    def score2(self, counts: collections.Counter) -> float:
        """
        使用Hellinger保真度评估模拟质量
        
        Parameters:
        -----------
        counts : Counter
            实验测量结果
            
        Returns:
        --------
        float : Hellinger保真度 [0,1]
        """
        # 获取电路并移除测量
        qc = self.circuit()
        qc_no_measure = qc.copy()
        qc_no_measure.data = [
            inst for inst in qc_no_measure.data 
            if inst[0].name != 'measure'
        ]

        # 生成理想状态向量
        sv = Statevector.from_instruction(qc_no_measure)
        ideal_probs = sv.probabilities_dict()

        # 归一化实验概率
        total_shots = sum(counts.values())
        exp_probs = {k: v / total_shots for k, v in counts.items()}

        # 计算Hellinger保真度
        fidelity = sum(
            np.sqrt(ideal_probs.get(k, 0)) * np.sqrt(p) 
            for k, p in exp_probs.items()
        )

        return fidelity


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """
    统计电路中两比特门的数量
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        量子电路
        
    Returns:
    --------
    int : 两比特门数量
    """
    dag = circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")
    return len(list(dag.two_qubit_ops()))


def setup_qpugroup(qpu_configs: List[Tuple[int, str]], 
                   connections: List[Tuple[int, int]], 
                   dist: float = 0.5) -> QPUManager:
    """
    设置QPU组
    
    Parameters:
    -----------
    qpu_configs : List[Tuple[int, str]]
        QPU配置列表，每个元素为(qpu_id, backend_name)
    connections : List[Tuple[int, int]]
        QPU连接列表，每个元素为(qpu1_id, qpu2_id)
    dist : float
        QPU间距离，默认0.5
        
    Returns:
    --------
    QPUManager : 配置好的QPU管理器
    """
    qpugroup = QPUManager()
    
    # 添加QPU
    for qpu_id, backend_name in qpu_configs:
        qpugroup.add_qpu(DQCQPU(qpu_id, backend_name))
    
    # 添加连接
    for qpu1, qpu2 in connections:
        qpugroup.add_coonnection(qpu1, qpu2, distance=dist)
    
    return qpugroup


def run_hamiltonian_dqc(ham: HamiltonianSimulationQiskit, 
                        partition: List[int], 
                        qpugroup: QPUManager, 
                        numbits: int = 12) -> Tuple[float, int, int]:
    """
    运行哈密顿量DQC模拟并测量性能指标
    
    Parameters:
    -----------
    ham : HamiltonianSimulationQiskit
        哈密顿量模拟对象
    partition : List[int]
        量子比特分区方案
    qpugroup : QPUManager
        QPU管理器
    numbits : int
        保留的比特数，默认12
        
    Returns:
    --------
    Tuple[float, int, int] : (保真度, 远程门数量, 两比特门总数)
    """
    qc = ham.circuit()
    
    print(f"  电路信息: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
    
    try:
        # 执行DQC
        qc_dqc = DQCCircuit(qc)
        result_qc = qc_dqc.Execution(partition, qpugroup, comm_noise=True, measure_time = True)
        
        # 获取噪声模型并运行仿真
        noise_model = qc_dqc.get_noise_model()
        sim = AerSimulator(noise_model=noise_model)
        compiled = transpile(result_qc, sim)
        start_time = time.time()
        job = sim.run(compiled, shots=1000)
        result = job.result()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Running time: {elapsed_time:.4f} seconds")
        
        # 处理测量结果
        counts = result.get_counts()
        
        # 保留前numbits位
        new_counts = {}
        for bitstring, cnt in counts.items():
            bits = bitstring[:numbits]
            new_counts[bits] = new_counts.get(bits, 0) + cnt
        
        # 计算保真度 (Hellinger fidelity)
        fidelity = ham.score2(new_counts)
        
        # 获取性能指标
        num_remote_gates = qc_dqc.Num_RemoteGate
        num_two_qubit_gates = count_two_qubit_gates(result_qc)
        
        return fidelity, num_remote_gates, num_two_qubit_gates
    
    except Exception as e:
        print(f"  Error with partition {partition}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    """
    主函数 - 演示如何使用
    """
    # 创建12量子比特的TFIM电路
    ham = HamiltonianSimulationQiskit(num_qubits=20, time_step=1, total_time=3)
    qc = ham.circuit()
    qc0 = qc.copy()
    
    # 设置分布式量子计算
    qc_dqc = DQCCircuit(qc0)
    partition = [4, 4, 4, 4, 4]  # 4个QPU，每个3个量子比特
    
    # 配置QPU组
    qpugroup = QPUManager()
    for i in range(5):
        qpugroup.add_qpu(DQCQPU(i, "FakeVigoV2"))
 
    # 添加QPU连接（线性拓扑）
    distance = 5
    qpugroup.add_coonnection(0, 1, distance=distance)
    # 可以添加更多连接以形成不同的拓扑结构
    # qpugroup.add_coonnection(0, 2, distance=distance)
    qpugroup.add_coonnection(1, 2, distance=distance)
    # qpugroup.add_coonnection(1, 3, distance=distance)
    qpugroup.add_coonnection(2, 3, distance=distance)
    qpugroup.add_coonnection(3, 4, distance=distance)
    
    # 运行DQC模拟
    fidelity, num_remote, num_two_qubit = run_hamiltonian_dqc(
        ham, partition, qpugroup, numbits=20
    )
    
    print(f"\n结果:")
    print(f"  Hellinger保真度: {fidelity:.4f}")
    # print(f"  远程门数量: {num_remote}")
    # print(f"  两比特门总数: {num_two_qubit}")


if __name__ == "__main__":
    main()