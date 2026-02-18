from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit_aer import AerSimulator
from qiskit.circuit.library.standard_gates import HGate, XGate, ZGate, CXGate
from qiskit.circuit import Reset, Measure, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister
from qiskit.quantum_info import Kraus
from qiskit.converters import circuit_to_dag
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

# Backend IonQ
from .backend import IonQ

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error, pauli_error,
    amplitude_damping_error, phase_amplitude_damping_error,
    phase_damping_error
)

import numpy as np
import copy  
import time
import matplotlib.pyplot as plt

class RemoteGate(Instruction):
    """ Remote Gate"""
    def __init__(self, index: int, target: int):
        super().__init__("R", 1, 0, [])
        self.index = index
        self.target = target

class MX(Instruction):
    """MX for Control Side"""
    def __init__(self, index: int, target: int):
        super().__init__("MX", 2, 0, [])
        self.index = index
        self.target = target

class MZ(Instruction):
    """MZ for Target Side"""
    def __init__(self, index: int, target: int):
        super().__init__("MZ", 2, 0, [])
        self.index = index
        self.target = target

class AnsM(Instruction):
    """AnsM Measurement"""
    def __init__(self, mea: int):
        super().__init__("ANS_M", 1, 0, [])
        self.mea = mea

class S_CX(Instruction):
    """Custom CNOT Gate"""
    def __init__(self, control: int, target: int, path):
        super().__init__("S_CX", 2, 0, [])
        self.control = control
        self.target = target
        self.path = path

class MS(Instruction):
    """Multi-qubit Swap Gate"""
    def __init__(self, index: int, target: int):
        super().__init__("MS", 2, 0, [])
        self.index = index
        self.target = target

class IF_Z(Instruction):
    def __init__(self, index: int, target: int):
        super().__init__("IF_Z", 1, 0, [])
        self.index = index
        self.target = target

class IF_X(Instruction):
    """Conditional X Gate"""
    def __init__(self, index: int, target: int):
        super().__init__("IF_X", 1, 0, [])
        self.index = index
        self.target = target

# Mapping backend names to their classes
FAKE_BACKENDS = {
    "FakeVigoV2": FakeVigoV2,
    "FakeLagosV2": FakeLagosV2,
    "FakeCasablancaV2": FakeCasablancaV2,
    "FakeYorktownV2": FakeYorktownV2,
    "FakeManilaV2": FakeManilaV2,
    "FakeNairobiV2": FakeNairobiV2,
    "FakeMumbaiV2": FakeMumbaiV2,
    "FakeKolkataV2": FakeKolkataV2,
    "FakeGuadalupeV2": FakeGuadalupeV2,
    "FakeAlmadenV2": FakeAlmadenV2,
    "FakeAthensV2": FakeAthensV2,
    "FakeCambridgeV2": FakeCambridgeV2,
    "IonQ": IonQ
}

class QPUManager:
    def __init__(self):
        """
        Initialize QPU manager to handle multiple QPUs and their connections.
        
        Attributes:
            qpus: List of QPU instances
            noise_instructions: Dict mapping QPU pairs to noise instructions
            map: Adjacency list representing QPU network topology
            size: Total number of QPUs in the manager
        """
        # Store QPUs and noise instructions
        self.qpus = []
        self.noise_instructions = {}
        # Adjacency list: {qpu_id: [(neighbor_id, distance), ...]}
        self.map = {}  
        self.size = 0

    def add_qpu(self, qpu):
        """Add a QPU to the manager"""
        self.qpus.append(qpu)
        qpu_id = qpu.qpu_id
        self.size += 1
        # Initialize adjacency list entry if not exists
        if qpu_id not in self.map:
            self.map[qpu_id] = []

    def get_qpu(self, qpu_id):
        """Retrieve QPU instance by ID"""
        for qpu in self.qpus:
            if qpu.qpu_id == qpu_id:
                return qpu
        return None
    
    def add_coonnection(self, qpu_id1, qpu_id2, distance: float = 0):
        """Add bidirectional connection between two QPUs"""
        qpu1 = self.get_qpu(qpu_id1)
        qpu2 = self.get_qpu(qpu_id2)
        if qpu1 is None or qpu2 is None:
            raise ValueError(f"QPU {qpu_id1} or {qpu_id2} not found.")

        qpu1.add_connection(qpu_id2, distance)
        qpu2.add_connection(qpu_id1, distance)

        # Add to adjacency list if not already present
        if not any(n == qpu_id2 for n, _ in self.map[qpu_id1]):
            self.map[qpu_id1].append((qpu_id2, distance))
        if not any(n == qpu_id1 for n, _ in self.map[qpu_id2]):
            self.map[qpu_id2].append((qpu_id1, distance))

        # Store noise instructions for both directions
        noise_instr = qpu1.get_noise_by_distance(distance)
        self.noise_instructions[(qpu_id1, qpu_id2)] = noise_instr
        self.noise_instructions[(qpu_id2, qpu_id1)] = noise_instr

    def get_noise_instruction(self, qpu_id1, qpu_id2):
        """Get noise instruction for connection between two QPUs"""
        return self.noise_instructions.get((qpu_id1, qpu_id2), None)

    def check_connection(self, qpu_id1, qpu_id2) -> int:
        """Check if two QPUs are connected (returns 1 if connected, 0 otherwise)"""
        if qpu_id1 not in self.map:
            return 0
        return 1 if any(n == qpu_id2 for n, _ in self.map[qpu_id1]) else 0


class DQCQPU:
    def __init__(self, qpu_id: int, backend_name: str, connections=None, noise_type: str = None, **noise_kwargs):
        """
        Initialize a QPU instance with automatic custom instruction registration.

        :param qpu_id: QPU identifier
        :param backend_name: Name of the backend (e.g., "FakeLagosV2")
        :param connections: Optional list of connections to other QPUs
        :param noise_type: Type of noise model to apply
        :param noise_kwargs: Additional noise parameters
        """
        self.qpu_id = qpu_id

        # Validate backend name
        if backend_name not in FAKE_BACKENDS:
            raise ValueError(
                f"Unknown backend name '{backend_name}'. "
                f"Available options are: {list(FAKE_BACKENDS.keys())}"
            )

        # Initialize backend
        backend_cls = FAKE_BACKENDS[backend_name]
        backend = backend_cls()

        # Register custom instructions to target
        target = backend.target
        target.add_instruction(RemoteGate, name="R")
        target.add_instruction(MX, name="MX")
        target.add_instruction(MZ, name="MZ")
        target.add_instruction(AnsM, name="ANS_M")
        target.add_instruction(AnsM, name="IF_Z")
        target.add_instruction(AnsM, name="IF_X")
        target.add_instruction(S_CX, name="MS")

        self.backend = backend
        self.target = backend.target

        # Initialize connections and noise configuration
        self.connections = connections if connections else []
        
        if noise_type is not None:
            self.noise_config = {"type": noise_type, "params": noise_kwargs}
        else:
            self.noise_config = None

    def add_connection(self, other_qpu_id: int, distance: float = 1.0):
        """Add connection to another QPU with distance-based noise"""
        kraus = self.get_noise_by_distance(distance)
        conn = {
            "id": other_qpu_id,
            "distance": distance,
            "noise": kraus
        }
        self.connections.append(conn)

    def compile_x_gate(self):
        """Compile X gate for this QPU's backend"""
        qc = QuantumCircuit(1)
        qc.x(0)
        compiled = transpile(qc, self.backend)
        return compiled.data

    def compile_z_gate(self):
        """Compile Z gate for this QPU's backend"""
        qc = QuantumCircuit(1)
        qc.z(0)
        compiled = transpile(qc, self.backend)
        return compiled.data
    
    def exponential(self, L, alpha):
        """Calculate exponential decay based on distance"""
        return np.exp(-alpha * L)

    def get_noise_by_distance(self, distance=1):
        """
        Generate composite noise (amplitude damping + depolarizing) Kraus instruction
        based on transmission distance.
        :param distance: Transmission distance (e.g., km or m)
        :return: Qiskit Instruction containing Kraus operators
        """
        
        # Calculate amplitude damping parameter
        if distance == 0:
            gamma = 0
        else:
            gamma = self.exponential(distance, alpha=0.02)

        # Calculate depolarizing parameter
        p_depol = gamma

        gamma = 1 - gamma
        p_depol = 1 - p_depol

        # Amplitude damping Kraus operators
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        K_amp = [K0, K1]

        # Depolarizing Kraus operators
        sqrt1mp = np.sqrt(1 - p_depol)
        sqrt_p3 = np.sqrt(p_depol / 3)
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        K_depol = [sqrt1mp * I, sqrt_p3 * X, sqrt_p3 * Y, sqrt_p3 * Z]

        # Combine channels (amplitude → depolarizing)
        K_combined = [D @ A for D in K_depol for A in K_amp]

        # Create Qiskit Kraus instruction
        kraus_instr = Kraus(K_combined).to_instruction()
        
        return kraus_instr

    def __repr__(self):
        return (
            f"<QPU id={self.qpu_id}, size={self.size}, "
            f"backend={self.backend.name()}, max_qubits={self.backend.num_qubits}>"
        )

# -------------------------------------------------------------
class DQCCircuit(QuantumCircuit):
    def __init__(self, *args, **kwargs):
        #  用已有 QuantumCircuit 初始化
        if len(args) == 1 and isinstance(args[0], QuantumCircuit):
            qc = args[0]
            # 保留原寄存器结构
            super().__init__(*qc.qregs, *qc.cregs,
                             name=qc.name,
                             global_phase=qc.global_phase,
                             metadata=copy.deepcopy(qc.metadata) if qc.metadata else None)

            # 复制电路数据
            self.data = copy.deepcopy(qc.data)
        else:
            super().__init__(*args, **kwargs)

        self.step = []                              # 各阶段存储的电路图
        self.step.append(copy.deepcopy(self))

        self.sub_circuit = []                       # 子电路
        self.sub_circuit_trans = []                 # 编译后的子电路集合
        self.result_circuit = None                  # 最终结果电路

        self.partition = []                         # 划分
        self.qubit_group = [-1] * self.num_qubits   # 比特分组
        self.Entanglement_swapping = []             # 纠缠信息
        self.swap_routes = []                       # 交换路径

        self.qubit_tele = []                        # 比特对应的通信比特位置

        self.qpus = []
        self.merged_qubits_map = {}               # 合并后比特映射
        self.merged_qubits_map_reverse = {}         # 反向映射

        self.qpugroup = None

        self.Num_Entanglement_swapping = 0          # 纠缠交换次数统计
        self.Num_RemoteGate = 0

    # 执行
    def Execution(self, config, qpugroup, comm_noise=False, measure_time=False):
        # 开始计时
        if measure_time:
            start_time = time.time()
        
        self.qpugroup = qpugroup
        qpus = self.qpugroup.qpus

        self.split(config)
        self.valid_trans()
        self.check_swap_entanglement()
        print("Entanglement Number:", self.Num_Entanglement_swapping)
        self.rearrange_with_partition()
        self.rewrite_cross_group_cnots()
        self.physic_split()
        
        self.transpile_subcircuits(qpus)
        # for sub_circ in self.sub_circuit_trans:
        #     sub_circ.draw("mpl", scale = 0.5, fold = 100)
        #     plt.show()

        # dag = circuit_to_dag(self.sub_circuit_trans[2])
        # dag.draw(output='mpl')  
        # plt.show()

        result_qc = self.merge_trans_circuits(comm_noise)
        
        # 结束计时并输出
        if measure_time:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"SimDisQ Execution time: {elapsed_time:.4f} seconds")
        
        return result_qc

    # Get the index of a qubit in the circuit
    def get_index(self, q):
        return int(self.find_bit(q).index)

    # Split the circuit based on the provided configuration
    def split(self, config):
        partition = []

        if all(isinstance(x, int) for x in config):
            # 按长度生成索引列表
            start = 0
            for s in config:
                indices = list(range(start, start + s))
                partition.append(indices)
                start += s
        elif all(isinstance(x, (list, tuple)) for x in config):
            # 按指定索引组合, 并排序
            for sub in config:
                partition.append(sorted(sub))
        else:
            raise ValueError("Input must be a list of ints or a list of lists of ints")
        
        self.partition = partition

        for gid, group in enumerate(partition):
            self.Entanglement_swapping.append(0)
            for q in group:
                self.qubit_group[q] = gid
        return partition

    # Preprocessing to validate and decompose cross-group multi-qubit gates to single-qubit gates and CX gates
    # step[1]
    def valid_trans(self):
        """
        遍历 old_circ 的指令，检查是否跨组多比特门。
        如果跨组且不是 CX，则分解后 append 到 new_circ；
        如果同组或是 CX，则直接 append。
        
        参数:
            new_circ: DQCCircuit，目标电路
            old_circ: DQCCircuit，源电路（step[0]）
        """
        old_circ = self

         # === 1. 构建新电路（仅复制量子比特）===
        new_circ = DQCCircuit(old_circ.qubits)
        new_circ.qubit_group = old_circ.qubit_group
        new_circ.qubit_tele = old_circ.qubit_tele
        print("Qubit Group:", new_circ.qubit_group)

        # === 2. 继承所有经典寄存器 ===
        # 复制通信寄存器（如果有）
        comm_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" in creg.name]
        for comm_creg in comm_cregs:
            new_circ.add_register(ClassicalRegister(len(comm_creg), comm_creg.name))

        # 复制原始寄存器（如果有）
        orig_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" not in creg.name]
        for orig_creg in orig_cregs:
            new_circ.add_register(ClassicalRegister(len(orig_creg), orig_creg.name))

        for instri in old_circ.data:
            instr = instri.operation   # 量子门或操作对象
            qargs = instri.qubits      # 作用的量子比特列表
            cargs = instri.clbits      # 作用的经典比特列表
            
            # 先处理cx
            if instr.name in ["cx"]:
                new_circ.append(instr, qargs, cargs)
                continue

            # 单比特门直接 append
            if len(qargs) <= 1:
                new_circ.append(instr, qargs, cargs)
                continue
            
            # 多比特门，判断是否跨组
            groups = [new_circ.qubit_group[old_circ.get_index(q)] for q in qargs]
            if len(set(groups)) > 1:
                # 跨组且不是 CX，分解
                ci = CircuitInstruction(instr, qargs, cargs)
                decomposed_instrs = self.decompose_and_get_data(ci)
                for di in decomposed_instrs:
                    mapped_qubits = [new_circ.qubits[old_circ.get_index(q)] for q in di.qubits]
                    mapped_clbits = [new_circ.clbits[old_circ.get_index(c)] for c in di.clbits]
                    new_circ.append(di.operation, mapped_qubits, mapped_clbits)
            else:
                # 同组，多比特门直接 append
                new_circ.append(instr, qargs, cargs)
        
        self.step.append(new_circ)
        return new_circ
    
    # step[2]
    def check_swap_entanglement(self):
        """检查跨 QPU 纠缠，如果无直接连接则寻找可行的 SWAP 路径"""
        swap_routes = []
        qpu_map = self.qpugroup.map

        old_circ = self.step[1]

         # === 1. 构建新电路（仅复制量子比特）===
        new_circ = DQCCircuit(old_circ.qubits)

        for creg in getattr(old_circ, "cregs", []):
            new_circ.add_register(ClassicalRegister(len(creg), creg.name))

        # 遍历电路中的每一个 CX 操作
        for instr in old_circ.data:
            if instr.operation.name == "cx":
                ctrl, tgt = instr.qubits
                g_ctrl = self.qubit_group[self.get_index(ctrl)]
                g_tgt = self.qubit_group[self.get_index(tgt)]

                # 如果控制和目标在不同 QPU 上
                if g_ctrl != g_tgt:
                    # 检查是否直接连接
                    if not self.qpugroup.check_connection(g_ctrl, g_tgt):
                        # --- 没有直接连接：寻找最短路径 ---
                        path = self._find_shortest_path(qpu_map, g_ctrl, g_tgt)
                        if path:
                            swap_routes.append(path)
                            new_circ.append(S_CX(g_ctrl, g_tgt, path), instr.qubits, instr.clbits)
                            print(f"[Info] Found SWAP route {path} for CX({g_ctrl}, {g_tgt})")
                            self.Num_Entanglement_swapping += len(path) - 2
                        else:
                            print(f"[Warning] No route found between QPU {g_ctrl} and {g_tgt}")
                    else:
                        # 直接连接，直接添加 CX 操作
                        new_circ.append(instr.operation, instr.qubits, instr.clbits)
                else:
                    # 同组，直接添加 CX 操作
                    new_circ.append(instr.operation, instr.qubits, instr.clbits)        
            else:
                new_circ.append(instr.operation, instr.qubits, instr.clbits)

        for path in swap_routes:
            # 取除首尾外的中间节点
            for node in path[1:-1]:
                self.Entanglement_swapping[node] = 1
                
        self.swap_routes = swap_routes
        self.step.append(new_circ)

    def _find_shortest_path(self, graph, start, end):
        """使用 BFS 在邻接表图中寻找最短路径"""
        from collections import deque

        visited = set()
        queue = deque([[start]])  # 队列中每个元素是路径 list

        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end:
                return path  # 找到路径

            if node not in visited:
                visited.add(node)
                for neighbor, _ in graph.get(node, []):  # 遍历邻居
                    if neighbor not in visited:
                        queue.append(path + [neighbor])

        return None  # 无路径                     

    # step[3]
    def rearrange_with_partition(self):
        """
        根据 partition 重新排列 qubits, 每组加一个通信比特,
        并为 group 之间的通信分配 classical bits。
        """
        if not self.partition:
            raise ValueError("请先设置 self.partition")

        partition = self.partition
        num_groups = len(partition)

        # classical bit 数量 = g * (g-1)
        comm_creg = ClassicalRegister(num_groups * (num_groups - 1), "Tele")

        # 每组 qubits 数量 = 原本 + 1(comm)

        # 先统计 partition 中普通 qubit 数量
        total_qubits = sum(len(group) for group in partition)
        # 给每组添加一个通信 qubit
        total_qubits += len(partition)
        # 给每个需要交换纠缠的 QPU 添加一个通信 qubit
        total_qubits += self.Entanglement_swapping.count(1)

        new_qreg = QuantumRegister(total_qubits, "q")

        new_circ = DQCCircuit(new_qreg, comm_creg)

        old_circ = self.step[2]
        # === 4. 保留原始经典寄存器信息 ===
        # 如果原始电路 self 中有经典寄存器
        if hasattr( old_circ, "clbits") and old_circ.clbits:
            # 尝试从 self.cregs 中找到原寄存器名
            if hasattr( old_circ, "cregs") and old_circ.cregs:
                # 取第一个 ClassicalRegister 的名字
                orig_name =  old_circ.cregs[0].name
            else:
                orig_name = "c"  # 如果没有记录，则默认命名为 "c"

            # 使用原寄存器名创建新的 ClassicalRegister
            orig_creg = ClassicalRegister(len(old_circ.clbits), orig_name)
            new_circ.add_register(orig_creg)

            # 保存引用方便之后访问
            new_circ.orig_creg = orig_creg
        else:
            new_circ.orig_creg = None

        # ===== 建立旧 qubit -> 新 qubit 映射 =====
        old2new = {}
        group_comm_qubits = []
        new_index = 0

        for indices in partition:
            # 字典映射量子寄存器
            # group 内 qubit
            for qi in indices:
                old2new[qi] = new_qreg[new_index]   
                new_index += 1
            # comm qubit
            comm_q = new_qreg[new_index]
            group_comm_qubits.append(comm_q)
            new_index += 1
            if self.Entanglement_swapping[self.qubit_group[indices[0]]] == 1:
                # 需要交换纠缠的 QPU 多加一个 comm qubit
                extra_comm_q = new_qreg[new_index]
                group_comm_qubits.append(extra_comm_q)
                new_index += 1

        # ===== 遍历原电路, 重映射到新电路 =====
        for instr in old_circ.data:
            qubit_indices = [old_circ.get_index(q) for q in instr.qubits]

            if all(qi in old2new for qi in qubit_indices):
                new_qargs = [old2new[qi] for qi in qubit_indices]
                new_circ.append(instr.operation, new_qargs, instr.clbits)

        # ==== 给 new_circ.qubit_group 填值 =====
        new_circ.qubit_group = []
        for gid, group in enumerate(partition):
            base_count = len(group) + 1  # 普通 + 通信
            if self.Entanglement_swapping[gid] == 1:
                base_count += 1  # 额外加一个 SWAP qubit
            new_circ.qubit_group.extend([gid] * base_count)

        # ===== 给 new_circ.qubit_tele 填值 =====
        qubit_tele = [-1] * len(new_qreg) 

        idx = 0
        for gid, group in enumerate(partition):
            # 每个分区的普通 + 通信 qubit 数量
            comm_q = group_comm_qubits[idx]
            comm_idx = new_circ.get_index(comm_q)

            # 普通 qubit 指向该组通信 qubit
            for qi in group:
                qubit_tele[new_circ.get_index(old2new[qi])] = comm_idx

            # 通信 qubit 自己设为 -1
            qubit_tele[comm_idx] = -1
            idx += 1

            # 若该组需要 entanglement swapping，则还要处理多出来的通信 qubit
            if self.Entanglement_swapping[gid] == 1:
                swap_comm_q = group_comm_qubits[idx]
                swap_comm_idx = new_circ.get_index(swap_comm_q)
                qubit_tele[swap_comm_idx] = -1  # 交换通信 qubit 也设为 -1
                idx += 1

        # 赋值
        new_circ.qubit_tele = qubit_tele

        self.step.append(new_circ)
        return new_circ

    # Rewrite cross-group CNOTs into RemoteGate, Measurement, If_X, If_Z
    # step[4]
    def rewrite_cross_group_cnots(self):
        
        old_circ = self.step[3]

        # === 1. 构建新电路（仅复制量子比特）===
        new_circ = DQCCircuit(old_circ.qubits)
        new_circ.qubit_group = old_circ.qubit_group
        new_circ.qubit_tele = old_circ.qubit_tele
        group_tele = {gid: self.step[3].qubit_tele[self.step[3].qubit_group.index(gid)] for gid in set(self.step[3].qubit_group)}

        # === 2. 继承所有经典寄存器 ===
        # 复制通信寄存器（如果有）
        comm_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" in creg.name]
        for comm_creg in comm_cregs:
            new_circ.add_register(ClassicalRegister(len(comm_creg), comm_creg.name))

        # 复制原始寄存器（如果有）
        orig_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" not in creg.name]
        for orig_creg in orig_cregs:
            new_circ.add_register(ClassicalRegister(len(orig_creg), orig_creg.name))


        idx_counter = 0  # 全局 index 计数器

        # === 4. 改写电路 ===
        for inst_obj in old_circ.data:
            instr = inst_obj.operation
            qargs = inst_obj.qubits
            cargs = inst_obj.clbits
            if instr.name == "cx": 
                ctrl, tgt = qargs
                g_ctrl = new_circ.qubit_group[old_circ.get_index(ctrl)]
                g_tgt = new_circ.qubit_group[old_circ.get_index(tgt)]

                if g_ctrl != g_tgt:
                    # ====== 跨组 CNOT，改写 ======
                    ctrl_index = old_circ.get_index(ctrl)
                    tgt_index = old_circ.get_index(tgt)
                    ctrl_comm_index = old_circ.qubit_tele[ctrl_index]
                    tgt_comm_index = old_circ.qubit_tele[tgt_index]

                    ctrl_comm = new_circ.qubits[ctrl_comm_index]
                    tgt_comm = new_circ.qubits[tgt_comm_index]
                    ctrl_q = new_circ.qubits[ctrl_index]
                    tgt_q = new_circ.qubits[tgt_index]

                    # Reset 
                    new_circ.append(Reset(), [ctrl_comm])
                    new_circ.append(Reset(), [tgt_comm])

                    # RemoteGate
                    new_circ.append(RemoteGate(idx_counter, g_tgt), [ctrl_comm])
                    new_circ.append(RemoteGate(idx_counter, g_ctrl), [tgt_comm])
                    idx_counter += 1

                    # CNOT: ctrl->ctrl_comm, tgt_comm->tgt
                    new_circ.append(CXGate(), [ctrl_q, ctrl_comm])
                    new_circ.append(CXGate(), [tgt_comm, tgt_q])

                    # H on tgt_comm
                    new_circ.append(HGate(), [tgt_comm])

                    # Measurement + If_Z
                    # Measurement + If_X
                    mz_inst = MZ(index=idx_counter, target=g_tgt)
                    mx_inst = MX(index=idx_counter, target=g_ctrl)
                    new_circ.append(mz_inst, [ctrl_q, ctrl_comm])
                    new_circ.append(mx_inst, [tgt_comm, tgt_q])
                    idx_counter += 1

                    self.Num_RemoteGate += 1
                else:
                    # ====== 同组 CNOT，保持原样 ======
                    new_circ.append(instr, qargs, cargs)
            elif instr.name == "measure":
                # ====== 测量指令替换为 AnsM ======
                qarg = qargs[0]
                carg = cargs[0] if cargs else None

                # 获取经典比特在其寄存器中的索引
                if carg is not None:
                    mea_index = old_circ.find_bit(carg).index
                else:
                    mea_index = -1  # 没有关联经典比特

                # 创建 AnsM 占位指令
                ansm_gate = AnsM(mea_index)

                # 添加到新电路
                new_circ.append(ansm_gate, [qarg])
            elif instr.name == "S_CX":
                path = instr.path
                ctrl, tgt = qargs
                g_ctrl = new_circ.qubit_group[old_circ.get_index(ctrl)]
                g_tgt = new_circ.qubit_group[old_circ.get_index(tgt)]

                ctrl_index = old_circ.get_index(ctrl)
                tgt_index = old_circ.get_index(tgt)
                ctrl_comm_index = old_circ.qubit_tele[ctrl_index]
                tgt_comm_index = old_circ.qubit_tele[tgt_index]

                ctrl_q = new_circ.qubits[ctrl_index]
                tgt_q = new_circ.qubits[tgt_index]
                ctrl_comm = new_circ.qubits[ctrl_comm_index]
                tgt_comm = new_circ.qubits[tgt_comm_index]

                new_circ.append(Reset(), [ctrl_comm_index])

                for i in range(1, len(path) - 1):
                    mid_g = path[i]
                    tgt_g = path[i + 1]

                    mid_comm_index_1 = group_tele[mid_g]
                    mid_comm_index_2 = mid_comm_index_1 + 1
                    tgt_comm_index = group_tele[tgt_g]

                    mid_comm_1 = new_circ.qubits[mid_comm_index_1]
                    mid_comm_2 = new_circ.qubits[mid_comm_index_2]
                    tgt_comm = new_circ.qubits[tgt_comm_index]

                    if i == 1:
                        # Reset 
                        new_circ.append(Reset(), [mid_comm_1])
                        new_circ.append(Reset(), [mid_comm_2])
                        new_circ.append(Reset(), [tgt_comm])

                        # RemoteGate
                        new_circ.append(RemoteGate(idx_counter, mid_g), [ctrl_comm])
                        new_circ.append(RemoteGate(idx_counter, g_ctrl), [mid_comm_1])
                        idx_counter += 1

                        new_circ.append(RemoteGate(idx_counter, tgt_g), [mid_comm_2])
                        new_circ.append(RemoteGate(idx_counter, mid_g), [tgt_comm])
                        idx_counter += 1
                        
                        new_circ.append(CXGate(), [mid_comm_1, mid_comm_2])
                        new_circ.append(HGate(), [mid_comm_1])

                        new_circ.append(IF_Z(index=idx_counter, target=mid_g), [ctrl_comm_index])
                        new_circ.append(MS(index=idx_counter, target=tgt_g), [mid_comm_1, mid_comm_2])
                        new_circ.append(IF_X(index=idx_counter, target=g_ctrl), [tgt_comm_index])
                        idx_counter += 1
                    else:
                        new_circ.append(Reset(), [mid_comm_2])
                        new_circ.append(Reset(), [tgt_comm])

                        # RemoteGate
                        new_circ.append(RemoteGate(idx_counter, tgt_g), [mid_comm_2])
                        new_circ.append(RemoteGate(idx_counter, mid_g), [tgt_comm])
                        idx_counter += 1
                        
                        new_circ.append(CXGate(), [mid_comm_1, mid_comm_2])
                        new_circ.append(HGate(), [mid_comm_1])

                        new_circ.append(IF_Z(index=idx_counter, target=mid_g), [ctrl_comm_index])
                        new_circ.append(MS(index=idx_counter, target=tgt_g), [mid_comm_1, mid_comm_2])
                        new_circ.append(IF_X(index=idx_counter, target=g_ctrl), [tgt_comm_index])
                        idx_counter += 1             

                
                # CNOT: ctrl->ctrl_comm, tgt_comm->tgt
                new_circ.append(CXGate(), [ctrl_q, ctrl_comm])
                new_circ.append(CXGate(), [tgt_comm, tgt_q])

                # H on tgt_comm
                new_circ.append(HGate(), [tgt_comm])

                # Measurement + If_Z
                # Measurement + If_X
                mz_inst = MZ(index=idx_counter, target=g_tgt)
                mx_inst = MX(index=idx_counter, target=g_ctrl)
                new_circ.append(mz_inst, [ctrl_q, ctrl_comm])
                new_circ.append(mx_inst, [tgt_comm, tgt_q])
                idx_counter += 1   
            else:
                # 不是 CNOT 和 Mesurement，保持原样
                new_circ.append(instr, qargs, cargs)

        self.step.append(new_circ)
        return new_circ

    # Physically split the circuit into sub-circuits based on partition
    def physic_split(self):
        config = [len(group) + 1 for group in self.partition]
        for gid, flag in enumerate(self.Entanglement_swapping):
            if flag == 1:
                config[gid] += 1

        temp_circuit = self.step[4]
        sub_circuits = []

        if all(isinstance(x, int) for x in config):
            # 按数量顺序分割, 例如 [2,2,3]
            start = 0
            for s in config:
                sub = DQCCircuit(s)
                sub.qubit_group = temp_circuit.qubit_group[start:start+s]

                # 给子电路加 classical bits
                num_clbits = (len(config) - 1) * 2
                if num_clbits > 0:
                    creg = ClassicalRegister(num_clbits, "c")
                    sub.add_register(creg)

                # 复制原电路对应 qubit 的操作
                for instr in temp_circuit.data:
                    qubit_indices = [temp_circuit.get_index(q) for q in instr.qubits]
                    # 只保留属于子电路的 qubit
                    indices_in_sub = [i for i, qi in enumerate(qubit_indices) if start <= qi < start+s]
                    if indices_in_sub:
                        new_qargs = [sub.qubits[qi - start] for i, qi in enumerate(qubit_indices) if i in indices_in_sub]
                        new_cargs = []
                        # print(f"Appending instruction {instr.operation.name} to sub-circuit with qubits {new_qargs}")
                        sub.append(instr.operation, new_qargs, new_cargs)
                        
                sub_circuits.append(sub)
                start += s

        elif all(isinstance(x, (list, tuple)) for x in config):
            # 按指定索引组合, 例如 [[1,3],[0,2,4]]
            for indices in config:
                sub = DQCCircuit(len(indices))
                sub.qubit_type = [self.qubit_type[i] for i in indices]

                for instr in self.data:
                    qubit_indices = [self.get_index(q) for q in instr.qubits]
                    # 只保留在 indices 中的 qubit
                    indices_in_sub = [i for i, qi in enumerate(qubit_indices) if qi in indices]
                    if indices_in_sub:
                        new_qargs = [sub.qubits[indices.index(qi)] for i, qi in enumerate(qubit_indices) if i in indices_in_sub]
                        new_cargs = []
                        sub.append(instr.operation, new_qargs, new_cargs)

                sub_circuits.append(sub)

        else:
            raise ValueError("Config must be a list of ints or a list of lists of ints")

        self.sub_circuit = sub_circuits
        return sub_circuits

    # Protect custom instructions with barriers
    def protect_custom_instructions(self, subcirc):
        """
        为 RemoteGate、Measurement、If_X、If_Z、AnsM 添加 barrier 保护。
        """
        new_circ = DQCCircuit(*subcirc.qregs, *subcirc.cregs)
        
        for inst_obj in subcirc.data:
            inst = inst_obj.operation
            qargs = inst_obj.qubits
            cargs = inst_obj.clbits

            if isinstance(inst, (MX, MZ, AnsM, IF_Z, IF_X, MS, S_CX)):               # 在同样的 qubits 上加 barrier
                new_circ.barrier(*qargs)
                new_circ.append(inst, qargs, cargs)
                new_circ.barrier(*qargs)
                # print(f"[Info] Protecting instruction: {inst.name} on qubits {[q for q in qargs]}")
            elif isinstance(inst, (RemoteGate)):
                new_circ.barrier()
                new_circ.append(inst, qargs, cargs)
                new_circ.barrier()
            else:
                new_circ.append(inst, qargs, cargs)

        return new_circ

    # Restore custom instructions by removing barriers
    def restore_custom_instructions(self, subcirc):
        """
        还原 RemoteGate、Measurement、If_X、If_Z、AnsM 的 barrier 保护。
        """
        new_circ = DQCCircuit(*subcirc.qregs, *subcirc.cregs)
        
        for inst_obj in subcirc.data:
            inst = inst_obj.operation
            qargs = inst_obj.qubits
            cargs = inst_obj.clbits

            if inst.name != "barrier":
                new_circ.append(inst, qargs, cargs)
            # if isinstance(inst, (RemoteGate, MX, MZ, AnsM)):  
            #     print(f"[Info] Restoring instruction: {inst.name} on qubits {[q for q in qargs]}")
        return new_circ
    
    def transpile_subcircuits(self, qpus, layout_out=None):
        """
        Transpile each sub-circuit using the corresponding QPU backend target,
        and optional layout mapping per subcircuit.

        :param qpus: list of QPU instances (each having .backend and .target)
        :param layout_out: list of layouts; each layout is a list of physical qubit indices
                        corresponding to the sub-circuit's logical qubits
                        e.g. [[0,1,2,3], [5,6,7,8]]
        """
        # ---- 按 qpu_id 排序 ----
        self.qpus = sorted(qpus, key=lambda x: x.qpu_id)

        if not self.sub_circuit:
            raise ValueError("No sub-circuits found. Please populate self.sub_circuit first.")
        if len(qpus) != len(self.sub_circuit):
            raise ValueError(
                f"Number of QPUs ({len(qpus)}) does not match number of sub-circuits ({len(self.sub_circuit)})."
            )

        # ---- 检查 layout_out ----
        if layout_out is not None:
            if len(layout_out) != len(self.sub_circuit):
                raise ValueError("layout_out length must match number of sub-circuits.")

        # ---- 检查子电路规模是否符合后端限制 ----
        for idx, (sub_circ, qpu) in enumerate(zip(self.sub_circuit, qpus)):
            backend = qpu.backend
            size = sub_circ.num_qubits
            qpu_id = getattr(qpu, "qpu_id", idx)
            backend_name = getattr(backend, "name", "unknown")

            if size > backend.num_qubits:
                raise ValueError(
                    f"QPU {qpu_id}: requested size={size} exceeds "
                    f"the maximum number of qubits ({backend.num_qubits}) "
                    f"supported by backend {backend_name}."
                )

        self.sub_circuit_trans = []

        # ---- 对每个子电路执行 transpile ----
        for idx, (sub, qpu) in enumerate(zip(self.sub_circuit, qpus)):
            try:
                # Step 1: 保护自定义指令
                sub = self.protect_custom_instructions(sub)

                layout = None
                if layout_out is not None and idx < len(layout_out):
                    layout = layout_out[idx]

                # Step 3: 调用 transpile
                if layout is not None:
                    sub = transpile(
                        sub,
                        backend=qpu.backend,
                        initial_layout=layout,
                        optimization_level=3,
                    )
                else:
                    sub = transpile(
                        sub,
                        backend=qpu.backend,
                        optimization_level=3,
                    )

                # Step 4: 还原自定义指令
                sub = self.restore_custom_instructions(sub)

                self.sub_circuit_trans.append(sub)
                print(f"[Info] Sub-circuit {idx} transpiled successfully on {qpu.backend.name}.")

            except Exception as e:
                print(f"[Error] Failed to transpile sub-circuit {idx} on {qpu.backend.name}: {e}")
                self.sub_circuit_trans.append(None)

        return self.sub_circuit_trans
    
    # Merge the transpiled sub-circuits into a complete circuit
    def merge_trans_circuits(self, comm_noise = False):
        """
        将 self.sub_circuit_trans 中的子电路组合成完整电路。
        支持跨电路配对操作 (R, M+IF_X/IF_Z), 通过栈实现返回上层逻辑。
        双向配对：无论先遇到 M 还是 IF_X/IF_Z 都能触发配对。
        """
        # === 构建 qubit / cbit 映射 ===
        qubits_map = {}
        merged_qubits_map = {}
        cbits_map = {}
        global_q_index = 0
        global_c_index = 0
        for i, sub in enumerate(self.sub_circuit_trans):
            if sub is None:
                continue
            local_indices = {sub.find_bit(q).index for instr in sub.data for q in instr.qubits}
            for local_index in sorted(local_indices):
                qubits_map[(i, local_index)] = global_q_index
                merged_qubits_map[global_q_index] = (i, local_index)
                global_q_index += 1

        # Record index for noise model
        self.merged_qubits_map = merged_qubits_map
        self.merged_qubits_map_reverse = qubits_map

        num_sub = len(self.sub_circuit_trans)
        for i in range(num_sub):
            for j in range(num_sub):
                if i != j:
                    cbits_map[(i, j)] = global_c_index
                    global_c_index += 1

        # === 初始化变量 ===
        # 创建新电路（只初始化量子比特）
        new_circ = DQCCircuit(len(qubits_map))
        new_circ.qubit_group = self.step[1].qubit_group

        # === 1. 添加通信寄存器 ===
        tele_creg = ClassicalRegister(len(cbits_map), "Tele")
        new_circ.add_register(tele_creg)

        # === 2. 保留原始经典寄存器 ===
        if hasattr(self, "clbits") and self.clbits:
            if hasattr(self, "cregs") and self.cregs:
                orig_name = self.cregs[0].name
            else:
                orig_name = "c"
            orig_creg = ClassicalRegister(len(self.clbits), orig_name)
            new_circ.add_register(orig_creg)

        indices = [0] * len(self.sub_circuit_trans)
        paired_op = {}      # {index: (sub_index, instr)}
        paired_op2 = {}      
        paired_done = set() # 已完成配对
        call_stack = []
        now = 0
        step = 0

        # 条件操作函数
        def apply_conditional_gate(gate_list, qubit, clbit):
            with new_circ.if_test((clbit, 1)):
                for instr in gate_list:
                    new_circ.append(instr.operation, [qubit], instr.clbits)

        while True:
            done = all(sub is None or indices[i] >= len(sub.data) for i, sub in enumerate(self.sub_circuit_trans))
            if done:
                break

            sub = self.sub_circuit_trans[now]
            if sub is None or indices[now] >= len(sub.data):
                if call_stack:
                    now = call_stack.pop()
                    continue
                else:
                    now = (now + 1) % len(self.sub_circuit_trans)
                    continue

            instr = sub.data[indices[now]]
            inst = instr.operation
            op_name = inst.name.upper()
            local_indices = [sub.find_bit(q).index for q in instr.qubits]
            global_qs = [qubits_map[(now, idx)] for idx in local_indices]
            target = getattr(inst, "target", None)
            idx = getattr(inst, "index", None)
            mea = getattr(inst,"mea", None)

            # print(f"\n[STEP] now={now}, target={target},idx={idx}, inst={inst.name}, indices={indices}")

            # === R 门配对生成 Bell 态 ===
            if op_name == "R" and idx not in paired_done:
                target_sub = self.sub_circuit_trans[target] if target is not None else None

                if idx not in paired_op:
                    paired_op[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                        continue
                else:
                    first_now = paired_op.pop(idx)
                    first_sub = self.sub_circuit_trans[first_now]
                    first_instr = first_sub.data[indices[first_now]]
                    first_qs = [qubits_map[(first_now, first_sub.find_bit(q).index)] for q in first_instr.qubits]
                    target_qs = global_qs
                    # print(first_now,now,first_qs,target_qs)
                    new_circ.initialize([1/np.sqrt(2),0,0,1/np.sqrt(2)], first_qs + target_qs)
                    if comm_noise:
                        noise_instr = self.qpugroup.get_noise_instruction(first_now, now)
                        # 插入噪声，映射到新电路 qubit
                        if isinstance(target_qs, list):
                            # 对列表中所有量子比特加噪声
                            noise_qargs = [new_circ.qubits[i] for i in target_qs]
                        else:
                            # 对单个 qubit 加噪声
                            noise_qargs = [new_circ.qubits[target_qs]]

                        if noise_instr is not None:
                            new_circ.append(noise_instr, noise_qargs)

                    paired_done.add(idx)
                    indices[now] += 1
                    indices[first_now] += 1
                    now = call_stack.pop() if call_stack else now
                    continue

            # === M / IF_X / IF_Z 双向配对 ===
            elif op_name in ("MX", "MZ") and idx not in paired_done:
                if idx not in paired_op:
                    paired_op[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                    continue

                # 已经配对
                first_now = paired_op.pop(idx)
                first_sub = self.sub_circuit_trans[first_now]
                first_instr = first_sub.data[indices[first_now]]

                first_qs = [qubits_map[(first_now, first_sub.find_bit(q).index)] 
                            for q in first_instr.qubits]
                target_qs = global_qs

                # classical bit 对应
                cbit_idx1 = cbits_map[(now, first_now)]
                cbit_idx2 = cbits_map[(first_now, now)]
                clbit_obj1 = new_circ.clbits[cbit_idx1]
                clbit_obj2 = new_circ.clbits[cbit_idx2]


                if first_instr.name == "MX":
                    new_circ.measure(first_qs[0], clbit_obj1)
                    apply_conditional_gate(self.qpus[now].compile_z_gate(), target_qs[0], clbit_obj1)
                    new_circ.measure(target_qs[1], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_x_gate(), first_qs[1], clbit_obj2)

                else:
                    new_circ.measure(first_qs[1], clbit_obj1)
                    apply_conditional_gate(self.qpus[now].compile_x_gate(), target_qs[1], clbit_obj1)
                    new_circ.measure(target_qs[0], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_z_gate(), first_qs[0], clbit_obj2)
                    

                # 更新状态
                paired_done.add(idx)
                indices[now] += 1
                indices[first_now] += 1
                now = call_stack.pop() if call_stack else now
                continue
            elif op_name == "ANS_M":
                # global_qs[0] 是量子比特索引，mea 是存储的经典比特索引
                q_idx = global_qs[0]
                c_idx = mea  # mea 已经是全局经典比特索引

                # 在 new_circ 中找到对应的 Clbit 对象
                clbit_obj = None
                temp_idx = c_idx
                for creg in new_circ.cregs:
                    if temp_idx < len(creg):
                        clbit_obj = creg[temp_idx]
                        break
                    else:
                        temp_idx -= len(creg)

                if clbit_obj is None:
                    raise ValueError(f"Cannot find the classical bit index {c_idx} corresponding to a Clbit")

                # 添加测量操作
                new_circ.measure(new_circ.qubits[q_idx], clbit_obj)
                indices[now] += 1
                continue
            elif op_name in ("IF_X", "IF_Z", "MS") and idx not in paired_done:
                target_sub = self.sub_circuit_trans[target] if target is not None else None
                if idx not in paired_op:
                    paired_op[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                    continue
                if idx not in paired_op2:
                    paired_op2[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                    continue

                # 已经配对
                first_now = paired_op.pop(idx)
                first_sub = self.sub_circuit_trans[first_now]
                first_instr = first_sub.data[indices[first_now]]

                second_now = paired_op2.pop(idx)
                second_sub = self.sub_circuit_trans[second_now]
                second_instr = second_sub.data[indices[second_now]]

                first_qs = [qubits_map[(first_now, first_sub.find_bit(q).index)] 
                            for q in first_instr.qubits]
                
                second_qs = [qubits_map[(second_now, second_sub.find_bit(q).index)] 
                            for q in second_instr.qubits]
                
                # instr global_qs
                third_qs = [qubits_map[(now, self.sub_circuit_trans[now].find_bit(q).index)] 
                            for q in instr.qubits]

                # classical bit 对应
                cbit_idx1 = cbits_map[(first_now, second_now)]  
                cbit_idx2 = cbits_map[(second_now, first_now)]  
                cbit_idx3 = cbits_map[(second_now, now)]        
                cbit_idx4 = cbits_map[(now, second_now)]                
                cbit_idx5 = cbits_map[(now, first_now)]         
                cbit_idx6 = cbits_map[(first_now, now)]         

                clbit_obj1 = new_circ.clbits[cbit_idx1]
                clbit_obj2 = new_circ.clbits[cbit_idx2] 
                clbit_obj3 = new_circ.clbits[cbit_idx3]
                clbit_obj4 = new_circ.clbits[cbit_idx4] 
                clbit_obj5 = new_circ.clbits[cbit_idx5]
                clbit_obj6 = new_circ.clbits[cbit_idx6] 
                # print("Entanglement Swapping", first_instr.name,second_instr.name, instr.name)
                if first_instr.name == "IF_Z":
                    new_circ.measure(second_qs[0], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_z_gate(), first_qs, clbit_obj2)
                    new_circ.measure(second_qs[1], clbit_obj3)
                    apply_conditional_gate(self.qpus[now].compile_x_gate(), third_qs, clbit_obj3)
                elif first_instr.name == "MS":
                    new_circ.measure(first_qs[0], clbit_obj6)
                    apply_conditional_gate(self.qpus[now].compile_z_gate(), third_qs, clbit_obj6)
                    new_circ.measure(first_qs[1], clbit_obj1)
                    apply_conditional_gate(self.qpus[second_now].compile_x_gate(), second_qs, clbit_obj1)
                elif first_instr.name == "IF_X":
                    new_circ.measure(third_qs[0], clbit_obj4)
                    apply_conditional_gate(self.qpus[second_now].compile_z_gate(), second_qs, clbit_obj4)
                    new_circ.measure(third_qs[1], clbit_obj5)
                    apply_conditional_gate(self.qpus[first_now].compile_x_gate(), first_qs, clbit_obj5)
                # 更新状态
                paired_done.add(idx)
                indices[now] += 1
                indices[first_now] += 1
                indices[second_now] += 1 
                now = call_stack.pop() if call_stack else now
                now = call_stack.pop() if call_stack else now
                continue
            # === 普通门 ===
            else:
                new_circ.append(inst, global_qs, [])
                indices[now] += 1
                continue

            step += 1
            if step > 50000:
                raise RuntimeError(f"[Error] Possible deadlock. indices={indices}, now={now}, stack={call_stack}")

        self.result_circuit = new_circ
        self.result_circuit.qubit_group = self.step[4].qubit_group
        return new_circ
    
    def decompose_and_get_data(self, instr: CircuitInstruction, basis_gates=None):
        """
        递归分解单个指令到基础门集。
        
        参数:
            instr: 待分解的 CircuitInstruction
            basis_gates: 允许的基础门名称集合（默认: {'cx'} + 单比特门）
        
        返回:
            保留原始 qubit 引用的 CircuitInstruction 列表（可直接用于 circuit.data）
        """
        if basis_gates is None:
            # 定义基础门集合
            basis_gates = {'cx', 'u3', 'u2', 'u1', 'id', 'x', 'y', 'z', 
                        'h', 's', 't', 'rx', 'ry', 'rz', 'sx', 'p'}
        
        # 如果是基础门或没有定义，直接返回
        if (not hasattr(instr.operation, "definition") or 
            instr.operation.name in basis_gates):
            return [instr]
        
        # 创建临时电路进行分解
        n_qubits = len(instr.qubits)
        qc_temp = QuantumCircuit(n_qubits)
        qc_temp.append(instr.operation, list(range(n_qubits)))
        
        # 建立临时 qubit 到原始 qubit 的映射
        # 临时电路的 qc_temp.qubits[i] 对应原始的 instr.qubits[i]
        temp_to_orig = {qc_temp.qubits[i]: instr.qubits[i] for i in range(n_qubits)}
        
        # 分解一次
        decomposed = qc_temp.decompose()
        
        result = []
        for sub_instr in decomposed.data:
            # 使用映射字典将临时 qubit 对象映射回原始 qubit 对象
            mapped_qubits = [temp_to_orig[q] for q in sub_instr.qubits]
            
            # 创建新的指令
            new_instr = CircuitInstruction(
                sub_instr.operation, 
                mapped_qubits, 
                sub_instr.clbits
            )
            
            # 递归分解（如果还需要）
            result.extend(self.decompose_and_get_data(new_instr, basis_gates))
        
        return result

    def reduce_noise_model(self, subset_qubits, noise_model=None, coupling_map=None):
        """
        裁剪 NoiseModel，使其只保留 subset_qubits 上的噪声，同时保留 basis_gates、description 和可选耦合图。
        
        参数:
            subset_qubits: list[int]
                需要保留噪声的量子比特
            noise_model: NoiseModel, 可选
                如果不提供，则使用 self.backend_noise_model
            coupling_map: list[tuple], 可选
                如果提供，则裁剪耦合图

        返回:
            NoiseModel
        """

        if noise_model is None:
            if getattr(self, "backend_noise_model", None) is None:
                raise ValueError("No noise model provided or set in self.backend_noise_model.")
            noise_model = self.backend_noise_model

        # 如果 Qiskit 版本支持 reduce()，优先使用
        if hasattr(noise_model, "reduce"):
            return noise_model.reduce(subset_qubits)

        sub_model = NoiseModel()

        # --- 1️⃣ 保留量子门噪声 ---
        for instr_name, qerrors in noise_model._local_quantum_errors.items():
            for qubits, error in qerrors.items():
                if all(q in subset_qubits for q in qubits):
                    sub_model.add_quantum_error(error, instr_name, qubits)

        # --- 2️⃣ 保留读出噪声 (measure) ---
        for qubit, error in noise_model._local_readout_errors.items():
            if qubit[0] in subset_qubits:  # qubit 是 tuple，如 (0,)
                sub_model.add_readout_error(error, [qubit[0]])  

        # --- 3️⃣ 保留 basis_gates 和 description ---
        if hasattr(noise_model, "basis_gates"):
            sub_model._basis_gates = list(noise_model.basis_gates)
        if hasattr(noise_model, "description"):
            sub_model._description = noise_model.description

        # --- 4️⃣ 可选裁剪耦合图 ---
        target_coupling_map = coupling_map if coupling_map is not None else getattr(self, "coupling_map", None)
        if target_coupling_map is not None:
            sub_model._coupling_map = [edge for edge in target_coupling_map if all(q in subset_qubits for q in edge)]
        else:
            sub_model._coupling_map = None

        return sub_model

    # Get a combined noise model for the distributed circuit based on QPU backends
    def get_noise_model(self):
        """
        构建一个综合噪声模型 (combined_noise_model)，
        将多个 QPU 的噪声模型裁剪后合并，并映射到全局连续 qubit。
        """
        # === Step 1: 参数检查 ===
        qpus = self.qpugroup.qpus
        qpus = sorted(qpus, key=lambda x: x.qpu_id)
        if not self.sub_circuit:
            raise ValueError("No sub-circuits found.")
        if len(qpus) != len(self.sub_circuit):
            raise ValueError(
                f"Number of QPUs ({len(qpus)}) does not match sub-circuits ({len(self.sub_circuit)})."
            )

        # === Step 2: 提取每个子电路对应的局部 qubit ===
        sub_qubit_maps = []
        for idx, _ in enumerate(self.sub_circuit):
            local_qubits = [
                local_index
                for global_q, (sub_index, local_index) in self.merged_qubits_map.items()
                if sub_index == idx
            ]
            sub_qubit_maps.append(local_qubits)

        # === Step 3: 构建全局 qubit 映射 (sub_idx, local_qubit) -> global_qubit ===
        qubit_mapping = self.merged_qubits_map_reverse

        # === Step 4: 初始化全局 NoiseModel ===
        combined_noise_model = NoiseModel()
        combined_basis_gates = set()

        # === Step 5: 对每个 QPU 裁剪并合并噪声 ===
        for idx, qpu in enumerate(qpus):
            backend_noise = NoiseModel.from_backend(qpu.backend)
            local_qubits = sub_qubit_maps[idx]

            # --- 5a. 裁剪局部噪声 ---
            reduced_noise = self.reduce_noise_model(
                subset_qubits=local_qubits,
                noise_model=backend_noise,
                coupling_map=qpu.backend.coupling_map
            )

            # --- 5b. 合并量子门噪声 ---
            for instr_name, qerrors in reduced_noise._local_quantum_errors.items():
                for qubits_tuple, error in qerrors.items():
                    # 解包 qubit
                    qubit_indices = [q if isinstance(q, int) else q[0] for q in qubits_tuple]
                    global_qubits = tuple(qubit_mapping[(idx, q)] for q in qubit_indices)
                    combined_noise_model.add_quantum_error(error, instr_name, global_qubits)
                    # print(f"Added quantum error: {instr_name}, local {qubit_indices} -> global {global_qubits}")

            # --- 5c. 合并读出噪声 ---
            for qubit, error in reduced_noise._local_readout_errors.items():
                q_local = qubit[0] if isinstance(qubit, tuple) else qubit
                if (idx, q_local) not in qubit_mapping:
                    print(f"⚠️ Warning: qubit mapping missing for {(idx, q_local)}")
                    continue
                global_qubit = qubit_mapping[(idx, q_local)]
                combined_noise_model.add_readout_error(error, [global_qubit])
                # print(f"Added readout error: local {q_local} -> global {global_qubit}")

            # --- 5d. 汇总 basis_gates ---
            if hasattr(reduced_noise, "basis_gates"):
                combined_basis_gates.update(reduced_noise.basis_gates)

        # === Step 6: 写入全局基础门、描述信息 ===
        combined_noise_model._basis_gates = list(combined_basis_gates)
        combined_noise_model._description = "Combined noise model from multiple QPUs"

        return combined_noise_model
    
def detect_swap_pattern(circuit):
    swap_count = 0
    instructions = list(circuit.data)
    
    i = 0
    while i < len(instructions) - 2:
        inst1 = instructions[i]
        inst2 = instructions[i + 1]
        inst3 = instructions[i + 2]
        
        # 检查是否都是 CX 门
        if (inst1.operation.name == 'cx' and 
            inst2.operation.name == 'cx' and 
            inst3.operation.name == 'cx'):
            
            # 获取量子比特索引
            q1_0 = circuit.find_bit(inst1.qubits[0]).index
            q1_1 = circuit.find_bit(inst1.qubits[1]).index
            
            q2_0 = circuit.find_bit(inst2.qubits[0]).index
            q2_1 = circuit.find_bit(inst2.qubits[1]).index
            
            q3_0 = circuit.find_bit(inst3.qubits[0]).index
            q3_1 = circuit.find_bit(inst3.qubits[1]).index
            
            # 检查 SWAP 模式: CX(a,b), CX(b,a), CX(a,b)
            if (q1_0 == q3_0 and q1_1 == q3_1 and  # 第1和第3个相同
                q2_0 == q1_1 and q2_1 == q1_0):     # 第2个是反向的
                swap_count += 1
                print(f"  检测到 SWAP 模式在位置 {i}: q{q1_0} ↔ q{q1_1}")
                i += 3  # 跳过这3个门
                continue
        
        i += 1
    
    print("\n" + "=" * 70)
    print("检测 CNOT 模式")
    print("=" * 70)
    print(f"检测到的 SWAP 数量: {swap_count}")
    
    return swap_count   