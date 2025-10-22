from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit_aer import AerSimulator
from qiskit.circuit.library.standard_gates import HGate, XGate, ZGate, CXGate
from qiskit.circuit import Reset, Measure, ClassicalRegister

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
    FakeAthensV2        # 5
)

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error, pauli_error,
    amplitude_damping_error, phase_amplitude_damping_error,
    phase_damping_error
)
from qiskit.quantum_info import Kraus

import numpy as np
import copy  

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

from qiskit import QuantumRegister

class RemoteGate(Instruction):
    """远程门 R, 占位符（单比特）"""
    def __init__(self, index: int, target: int):
        super().__init__("R", 1, 0, [])
        self.index = index
        self.target = target

class MX(Instruction):
    """远程测量 M, 占位符（单比特）
    index: 全局编号, 用于匹配 If_X 或 If_Z
    """
    def __init__(self, index: int, target: int):
        super().__init__("MX", 2, 0, [])
        self.index = index
        self.target = target

class MZ(Instruction):
    """远程测量 M, 占位符（单比特）
    index: 全局编号, 用于匹配 If_X 或 If_Z
    """
    def __init__(self, index: int, target: int):
        super().__init__("MZ", 2, 0, [])
        self.index = index
        self.target = target

class AnsM(Instruction):
    """占位符（单比特）"""
    def __init__(self, mea: int):
        super().__init__("ANS_M", 1, 0, [])
        self.mea = mea

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
}

class DQCQPU:
    def __init__(self, qpu_id: int, backend_name: str, connections=None, noise_type: str = None, **noise_kwargs):
        """
        Initialize a QPU instance with automatic custom instruction registration.

        :param qpu_id: QPU identifier
        :param size: Number of data qubits (excluding communication qubits)
        :param backend_name: Name of the backend (e.g., "FakeLagosV2")
        :param connections: Optional list of connections to other QPUs
        """
        self.qpu_id = qpu_id

        # ---- Validate backend name ----
        if backend_name not in FAKE_BACKENDS:
            raise ValueError(
                f"Unknown backend name '{backend_name}'. "
                f"Available options are: {list(FAKE_BACKENDS.keys())}"
            )

        # ---- Initialize backend ----
        backend_cls = FAKE_BACKENDS[backend_name]
        backend = backend_cls()

        # ---- Automatically register custom instructions ----
        target = backend.target
        target.add_instruction(RemoteGate, name="R")
        target.add_instruction(MX, name="MX")
        target.add_instruction(MZ, name="MZ")
        target.add_instruction(AnsM, name="ANS_M")

        self.backend = backend
        self.target = backend.target

        # ---- Initialize connections configuration ----
        self.connections = connections if connections else []
        # ---- Initialize noise configuration ----
        if noise_type is not None:
            self.noise_config = {"type": noise_type, "params": noise_kwargs}
        else:
            self.noise_config = None

    def add_connection(self, other_qpu_id: int, length: float = 1.0, noise_channel=None):
        """添加与其他QPU的连接, 附带延迟等信息"""
        conn = {
            "id": other_qpu_id,
            "length": length,
            "noise": noise_channel
        }
        self.connections.append(conn)

    def set_qubit_noise(self, noise_type: str, **kwargs):
        """保存指定 qubit 的噪声配置"""
        self.noise_config = {"type": noise_type, "params": kwargs}
    
    def compile_x_gate(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        compiled = transpile(qc, self.backend)
        return compiled.data  # 返回编译后指令对象列表

    def compile_z_gate(self):
        qc = QuantumCircuit(1)
        qc.z(0)
        compiled = transpile(qc, self.backend)
        return compiled.data  # 同样返回对象
    
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
        self.sub_circuit_trans_test = []                 # 编译后的子电路集合

        self.partition = []                         # 划分
        self.qubit_group = [-1] * self.num_qubits   # 比特分组

        self.qubit_tele = []                        # 比特对应的通信比特位置

        self.qpus = []

    # 执行
    def Execution(self, config, qpus):

        self.split(config)
        self.rearrange_with_partition()
        self.Valid_trans()
        self.rewrite_cross_group_cnots()
        self.physic_split()
        self.transpile_subcircuits(qpus)
        result_qc = self.merge_trans_circuits()

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
            for q in group:
                self.qubit_group[q] = gid

        return partition

    # Change the qubit arrangement based on partition
    # Add communication qubits for each group
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
        total_qubits = sum(len(group) + 1 for group in partition)
        new_qreg = QuantumRegister(total_qubits, "q")

        new_circ = DQCCircuit(new_qreg, comm_creg)

        # === 4. 保留原始经典寄存器信息 ===
        # 如果原始电路 self 中有经典寄存器
        if hasattr(self, "clbits") and self.clbits:
            # 尝试从 self.cregs 中找到原寄存器名
            if hasattr(self, "cregs") and self.cregs:
                # 取第一个 ClassicalRegister 的名字（一般只有一个）
                orig_name = self.cregs[0].name
            else:
                orig_name = "c"  # 如果没有记录，则默认命名为 "c"

            # 使用原寄存器名创建新的 ClassicalRegister
            orig_creg = ClassicalRegister(len(self.clbits), orig_name)
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

        # ===== 遍历原电路, 重映射到新电路 =====
        for instr in self.data:
            qubit_indices = [self.get_index(q) for q in instr.qubits]

            # 只映射 partition 覆盖到的 qubits
            if all(qi in old2new for qi in qubit_indices):
                new_qargs = [old2new[qi] for qi in qubit_indices]
                new_circ.append(instr.operation, new_qargs, instr.clbits)

        # ===== 给 new_circ 分组 =====
        new_circ.qubit_group = [
            gid for gid, group in enumerate(partition) for _ in range(len(group) + 1)
        ]

        # ===== 给 new_circ.qubit_tele 填值 =====
        qubit_tele = [-1] * len(new_qreg)  # 初始化，默认 -1
        for gid, group in enumerate(partition):
            comm_q = group_comm_qubits[gid]
            comm_idx = new_circ.get_index(comm_q)  # 通信 qubit 的索引
            for qi in group:
                qubit_tele[new_circ.get_index(old2new[qi])] = comm_idx
            # 自己通信 qubit 对应 -1
            qubit_tele[comm_idx] = -1

        new_circ.qubit_tele = qubit_tele

        self.step.append(new_circ)
        return new_circ

    # Preprocessing to validate and decompose cross-group multi-qubit gates to single-qubit gates and CX gates
    def Valid_trans(self):
        """
        遍历 old_circ 的指令，检查是否跨组多比特门。
        如果跨组且不是 CX，则分解后 append 到 new_circ；
        如果同组或是 CX，则直接 append。
        
        参数:
            new_circ: DQCCircuit，目标电路
            old_circ: DQCCircuit，源电路（step[0]）
        """
        old_circ = self.step[1]

         # === 1. 构建新电路（仅复制量子比特）===
        new_circ = DQCCircuit(old_circ.qubits)
        new_circ.qubit_group = old_circ.qubit_group

        # === 2. 继承所有经典寄存器 ===
        # 复制通信寄存器（如果有）
        comm_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" in creg.name]
        for comm_creg in comm_cregs:
            new_circ.add_register(ClassicalRegister(len(comm_creg), comm_creg.name))

        # 复制原始寄存器（如果有）
        orig_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" not in creg.name]
        for orig_creg in orig_cregs:
            new_circ.add_register(ClassicalRegister(len(orig_creg), orig_creg.name))

        # === 3. 初始化辅助属性 ===
        new_circ.qubit_tele = old_circ.qubit_tele

        qc = self.step[1]  # 原始电路
        for instri in qc.data:
            instr = instri.operation   # 量子门或操作对象
            qargs = instri.qubits      # 作用的量子比特列表
            cargs = instri.clbits      # 作用的经典比特列表
            
            # 单比特门直接 append
            if len(qargs) <= 1 or instr.name == "cx":
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

    # Rewrite cross-group CNOTs into RemoteGate, Measurement, If_X, If_Z
    def rewrite_cross_group_cnots(self):
        
        old_circ = self.step[2]

        # === 1. 构建新电路（仅复制量子比特）===
        new_circ = DQCCircuit(old_circ.qubits)
        new_circ.qubit_group = old_circ.qubit_group

        # === 2. 继承所有经典寄存器 ===
        # 复制通信寄存器（如果有）
        comm_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" in creg.name]
        for comm_creg in comm_cregs:
            new_circ.add_register(ClassicalRegister(len(comm_creg), comm_creg.name))

        # 复制原始寄存器（如果有）
        orig_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" not in creg.name]
        for orig_creg in orig_cregs:
            new_circ.add_register(ClassicalRegister(len(orig_creg), orig_creg.name))

        # === 3. 初始化辅助属性 ===
        new_circ.qubit_tele = old_circ.qubit_tele

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
            else:
                # 不是 CNOT 和 Mesurement，保持原样
                new_circ.append(instr, qargs, cargs)

        self.step.append(new_circ)
        return new_circ

    # Physically split the circuit into sub-circuits based on partition
    def physic_split(self):
        config = [len(i) + 1 for i in self.partition]
        temp_circuit = self.step[3]
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
        
        for inst, qargs, cargs in subcirc.data:
            if isinstance(inst, (RemoteGate, MX, MZ, AnsM)):               # 在同样的 qubits 上加 barrier
                new_circ.barrier(*qargs)
                new_circ.append(inst, qargs, cargs)
                new_circ.barrier(*qargs)
                # print(f"[Info] Protecting instruction: {inst.name} on qubits {[q for q in qargs]}")
            else:
                new_circ.append(inst, qargs, cargs)

        return new_circ

    # Restore custom instructions by removing barriers
    def restore_custom_instructions(self, subcirc):
        """
        还原 RemoteGate、Measurement、If_X、If_Z、AnsM 的 barrier 保护。
        """
        new_circ = DQCCircuit(*subcirc.qregs, *subcirc.cregs)
        
        for inst, qargs, cargs in subcirc.data:
            if inst.name != "barrier":
                new_circ.append(inst, qargs, cargs)
            # if isinstance(inst, (RemoteGate, MX, MZ, AnsM)):  
            #     print(f"[Info] Restoring instruction: {inst.name} on qubits {[q for q in qargs]}")
        return new_circ
    
    # Transpile each sub-circuit using the corresponding QPU backend target
    def transpile_subcircuits(self, qpus):
        """
        Transpile each sub-circuit using the corresponding QPU backend target.

        :param qpus: list of QPU instances (each having .backend.target)
        :raises ValueError: if sub_circuit count and qpu count mismatch
        """
        # ---- 按 qpu_id 排序 ----
        self.qpus = sorted(qpus, key=lambda x: x.qpu_id)

        if not self.sub_circuit:
            raise ValueError("No sub-circuits found. Please populate self.sub_circuit first.")
        if len(qpus) != len(self.sub_circuit):
            raise ValueError(
                f"Number of QPUs ({len(qpus)}) does not match number of sub-circuits ({len(self.sub_circuit)})."
            )
        # 检查子电路 qubit 数量是否超过 QPU 最大支持
        for idx, (sub_circ, qpu) in enumerate(zip(self.sub_circuit, qpus)):
            backend = qpu.backend
            size = sub_circ.num_qubits
            qpu_id = getattr(qpu, "qpu_id", idx)  # 使用 DQCQPU 的 qpu_id
            backend_name = getattr(backend, "name", "unknown")

            if size > backend.num_qubits:
                raise ValueError(
                    f"QPU {qpu_id}: requested size={size} exceeds "
                    f"the maximum number of qubits ({backend.num_qubits}) "
                    f"supported by backend {backend_name}."
                )
            
        self.sub_circuit_trans = [] 

        for idx, (sub, qpu) in enumerate(zip(self.sub_circuit, qpus)):
            try:
                # === Step 1: 保护自定义指令 ===
                sub = self.protect_custom_instructions(sub)
                #  sub.global_phase = 0  # 避免 global_phase 影响 transpile
                # === Step 2: 调用 transpile ===
                t_circ = transpile(sub, target=qpu.target, optimization_level=3)

                # === Step 3: 还原自定义指令 ===
                sub = self.restore_custom_instructions(sub)

                self.sub_circuit_trans.append(sub)
                print(f"[Info] Sub-circuit {idx} transpiled successfully on {qpu.backend.name}.")

            except Exception as e:
                print(f"[Error] Failed to transpile sub-circuit {idx} on {qpu.backend.name}: {e}")
                self.sub_circuit_trans.append(None)

        return self.sub_circuit_trans

    # Merge the transpiled sub-circuits into a complete circuit
    def merge_trans_circuits(self):
        """
        将 self.sub_circuit_trans 中的子电路组合成完整电路。
        支持跨电路配对操作 (R, M+IF_X/IF_Z), 通过栈实现返回上层逻辑。
        双向配对：无论先遇到 M 还是 IF_X/IF_Z 都能触发配对。
        """
        # === 构建 qubit / cbit 映射 ===
        qubits_map = {}
        cbits_map = {}
        global_q_index = 0
        global_c_index = 0
        for i, sub in enumerate(self.sub_circuit_trans):
            if sub is None:
                continue
            local_indices = {sub.find_bit(q).index for instr in sub.data for q in instr.qubits}
            for local_index in sorted(local_indices):
                qubits_map[(i, local_index)] = global_q_index
                global_q_index += 1

        num_sub = len(self.sub_circuit_trans)
        for i in range(num_sub):
            for j in range(num_sub):
                if i != j:
                    cbits_map[(i, j)] = global_c_index
                    global_c_index += 1

        # === 初始化变量 ===
        # 创建新电路（只初始化量子比特）
        new_circ = QuantumCircuit(len(qubits_map))

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
        paired_done = set() # 已完成配对
        paired_map = {}     # {index: global_qs} 保存第一次 M/IF_X/IF_Z 的 qubits
        call_stack = []
        ans_map = {}
        now = 0
        step = 0

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

            # print(f"\n[STEP] now={now}, inst={inst.name}, indices={indices}")

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
                    # print("first_qs:", first_qs)
                    # print("target_qs:", target_qs)
                    # print("overlap:", set(first_qs) & set(target_qs))
                    new_circ.initialize([1/np.sqrt(2),0,0,1/np.sqrt(2)], first_qs + target_qs)
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

                # 条件操作函数
                def apply_conditional_gate(gate_list, qubit, clbit):
                    with new_circ.if_test((clbit, 1)):
                        for instr in gate_list:
                            new_circ.append(instr.operation, [qubit], instr.clbits)

                if first_instr.name == "MX":
                    new_circ.measure(first_qs[0], clbit_obj1)
                    apply_conditional_gate(self.qpus[now].compile_z_gate(), target_qs[0], clbit_obj1)
                    new_circ.measure(target_qs[1], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_x_gate(), first_qs[1], clbit_obj2)
                    # print("Matched 1")
                else:
                    new_circ.measure(first_qs[1], clbit_obj1)
                    apply_conditional_gate(self.qpus[now].compile_x_gate(), target_qs[1], clbit_obj1)
                    new_circ.measure(target_qs[0], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_z_gate(), first_qs[0], clbit_obj2)
                    # print("Matched 2")

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
            # === 普通门 ===
            else:
                new_circ.append(inst, global_qs, [])
                indices[now] += 1
                continue

            step += 1
            if step > 50000:
                raise RuntimeError(f"[Error] Possible deadlock. indices={indices}, now={now}, stack={call_stack}")

        return new_circ
    
    # Split and decompose a single instruction based on given QPU targets
    def decompose_and_get_data(self, instr: CircuitInstruction):
        """
        对单个指令分解，保留 CX 和单比特门。
        返回 CircuitInstruction 列表，可直接 append 到电路中。
        保留原始 qubit 对象（来自 DQCCircuit）。
        """

        # 如果是基元门或 CX，直接返回
        if not hasattr(instr.operation, "definition") or instr.operation.name == "cx":
            return [instr]

        # 创建临时电路（仅用于分解，使用局部索引）
        temp_qubits = list(range(len(instr.qubits)))  # 临时量子比特索引
        qc_temp = QuantumCircuit(len(temp_qubits))
        qc_temp.append(instr.operation, temp_qubits)

        # 分解一次
        decomposed = qc_temp.decompose()

        result = []

        for i in decomposed.data:
            # 映射临时 qubit 回原始电路 qubit 对象
            mapped_qubits = [instr.qubits[idx] for idx, _ in enumerate(i.qubits)]

            if not hasattr(i.operation, "definition") or i.operation.name == "cx":
                result.append(CircuitInstruction(i.operation, mapped_qubits, i.clbits))
            else:
                # 复合门只分解一次
                sub_temp_qubits = list(range(len(i.qubits)))
                qc_sub = QuantumCircuit(len(sub_temp_qubits))
                qc_sub.append(i.operation, sub_temp_qubits)
                sub_decomposed = qc_sub.decompose()
                for sub in sub_decomposed.data:
                    mapped_sub_qubits = [instr.qubits[idx] for idx, _ in enumerate(sub.qubits)]
                    result.append(CircuitInstruction(sub.operation, mapped_sub_qubits, sub.clbits))

        return result


    # Get a combined noise model for the distributed circuit based on QPU backends
    # def get_noise_model(self, qpus):
    #     from qiskit_aer.noise import NoiseModel, depolarizing_error

    #     qpus = sorted(qpus, key=lambda x: x.qpu_id)

    #     if not self.sub_circuit:
    #         raise ValueError("No sub-circuits found.")
    #     if len(qpus) != len(self.sub_circuit):
    #         raise ValueError(
    #             f"Number of QPUs ({len(qpus)}) does not match sub-circuits ({len(self.sub_circuit)})."
    #         )

    #     combined_noise_model = NoiseModel()

    #     start = 0
    #     # ==== 每个 QPU 一段噪声 ====
    #     for i, qpu in enumerate(qpus):
    #         backend = qpu.backend

    #         num = len(self.sub_circuit[i].qubits)
    #         qubits_in_range = list(range(start, start + num))

    #         # 尝试从传入 backend 生成噪声
    #         try:
    #             backend_noise = NoiseModel.from_backend(backend)
    #             print(f"[Info] Using real noise model from {backend} for qubits {qubits_in_range}")
    #         except Exception:
    #             print(f"[Warning] Backend {backend} has no noise model, using default depolarizing noise.")
    #             backend_noise = None

    #         # ==== 若 backend 有噪声，取第一种单比特门噪声，否则自建 ====
    #         if backend_noise and backend_noise.to_dict().get("quantum_errors"):
    #             # 提取第一个单比特噪声模型作为代表
    #             first_error = None
    #             for qe in backend_noise.to_dict()["quantum_errors"]:
    #                 if len(qe["gate_qubits"][0]) == 1:  # 单比特门
    #                     first_error = backend_noise.get_quantum_error(qe["name"])
    #                     break
    #             if not first_error:
    #                 first_error = depolarizing_error(0.01, 1)
    #         else:
    #             first_error = depolarizing_error(0.01, 1)

    #         # ==== 对该 QPU 范围内的所有 qubit 添加噪声 ====
    #         single_qubit_gates = ['h', 'x', 'y', 'z', 'sx', 's', 't', 'rx', 'ry', 'rz']
    #         two_qubit_gates = ['cx']

    #         for g in single_qubit_gates:
    #             for q in qubits_in_range:
    #                 combined_noise_model.add_quantum_error(first_error, g, [q])

    #         # 给双比特门也加退极化噪声
    #         two_qubit_error = depolarizing_error(0.02, 2)
    #         for g in two_qubit_gates:
    #             for q in qubits_in_range[:-1]:
    #                 combined_noise_model.add_quantum_error(two_qubit_error, g, [q, q + 1])

    #         start += num

    #     return combined_noise_model



