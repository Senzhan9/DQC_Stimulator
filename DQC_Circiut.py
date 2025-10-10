from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Instruction
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

class Measurement(Instruction):
    """远程测量 M, 占位符（单比特）
    index: 全局编号, 用于匹配 If_X 或 If_Z
    """
    def __init__(self, index: int, target: int):
        super().__init__("M", 1, 0, [])
        self.index = index
        self.target = target

class If_X(Instruction):
    """远程 IF_X, 占位符（单比特）"""
    def __init__(self, index: int, target: int):
        super().__init__("IF_X", 1, 0, [])
        self.index = index
        self.target = target

class If_Z(Instruction):
    """远程 IF_Z, 占位符（单比特）"""
    def __init__(self, index: int, target: int):
        super().__init__("IF_Z", 1, 0, [])
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
        target.add_instruction(Measurement, name="M")
        target.add_instruction(If_X, name="IF_X")
        target.add_instruction(If_Z, name="IF_Z")
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

    def __repr__(self):
        return (
            f"<QPU id={self.qpu_id}, size={self.size}, "
            f"backend={self.backend.name()}, max_qubits={self.backend.num_qubits}>"
        )

# -------------------------------------------------------------
class DQCCircuit(QuantumCircuit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = []                              # 各阶段存储的电路图
        self.step.append(copy.deepcopy(self))

        self.sub_circuit = []                       # 子电路
        self.sub_circuit_trans = []                 # 编译后的子电路集合

        self.partition = []                         # 划分
        self.qubit_group = [-1] * self.num_qubits   # 比特分组

        self.qubit_tele = []                        # 比特对应的通信比特位置

    # 执行
    def Execution(self, config, qpus):

        self.split(config)
        self.rearrange_with_partition()
        self.rewrite_cross_group_cnots()
        self.physic_split()
        self.transpile_subcircuits(qpus)
        result_qc = self.merge_trans_circuits()

        return result_qc

    # 获取 Qubit 索引
    def get_index(self, q):
        return int(self.find_bit(q).index)

    # 根据传入的 config 整理划分信息并存入 partition 中
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

    def rewrite_cross_group_cnots(self):
        
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

                    # Measurement + If_X
                    new_circ.append(Measurement(idx_counter, g_tgt), [ctrl_comm])
                    new_circ.append(If_X(idx_counter, g_ctrl), [tgt_q])
                    idx_counter += 1

                    # Measurement + If_Z
                    new_circ.append(Measurement(idx_counter, g_ctrl), [tgt_comm])
                    new_circ.append(If_Z(idx_counter, g_tgt), [ctrl_q])
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

    # 将电路按照 partition 划分进 sub_circuit
    def physic_split(self):
        config = [len(i) + 1 for i in self.partition]
        temp_circuit = self.step[2]
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
    
    def transpile_subcircuits(self, qpus):
        """
        Transpile each sub-circuit using the corresponding QPU backend target.

        :param qpus: list of QPU instances (each having .backend.target)
        :raises ValueError: if sub_circuit count and qpu count mismatch
        """
        # ---- 按 qpu_id 排序 ----
        qpus = sorted(qpus, key=lambda x: x.qpu_id)

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
                t_circ = transpile(sub, target=qpu.target, optimization_level=3)
                self.sub_circuit_trans.append(t_circ)
                print(f"[Info] Sub-circuit {idx} transpiled successfully on {qpu.backend.name}.")
            except Exception as e:
                print(f"[Error] Failed to transpile sub-circuit {idx} on {qpu.backend.name}: {e}")
                self.sub_circuit_trans.append(None)

        return self.sub_circuit_trans

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
                    new_circ.initialize([1/np.sqrt(2),0,0,1/np.sqrt(2)], first_qs + target_qs)
                    paired_done.add(idx)
                    indices[now] += 1
                    indices[first_now] += 1
                    now = call_stack.pop() if call_stack else now
                    continue

            # === M / IF_X / IF_Z 双向配对 ===
            elif op_name in ("M", "IF_X", "IF_Z") and idx not in paired_done:
                target_sub = self.sub_circuit_trans[target] if target is not None else None

                if idx not in paired_op:
                    # 第一次遇到 M/IF_X/IF_Z
                    paired_op[idx] = (now, instr)
                    paired_map[idx] = global_qs
                    indices[now] += 1
                    if target_sub is not None:
                        call_stack.append(now)
                        now = target
                        continue
                else:
                    # 第二次遇到
                    first_now, first_instr = paired_op.pop(idx)
                    first_qs = paired_map[idx]
                    second_qs = global_qs
                    paired_done.add(idx)

                    first_name = first_instr.operation.name.upper()
                    second_name = op_name

                    # ------------------ 情况1：M 在前 ------------------
                    if first_name == "M":
                        if first_now != now:
                            cbit_idx = cbits_map[(first_now, now)]
                            clbit_obj = new_circ.clbits[cbit_idx]
                            new_circ.measure(first_qs[0], clbit_obj)
                            if second_name == "IF_X":
                                with new_circ.if_test((clbit_obj, 1)):
                                    new_circ.x(second_qs[0])
                            elif second_name == "IF_Z":
                                with new_circ.if_test((clbit_obj, 1)):
                                    new_circ.rz(np.pi / 2, second_qs[0])
                        else:
                            new_circ.measure(first_qs[0], first_qs[0])
                            if second_name == "IF_X":
                                new_circ.x(second_qs[0])
                            elif second_name == "IF_Z":
                                new_circ.rz(np.pi / 2, second_qs[0])

                    # ------------------ 情况2：IF_X / IF_Z 在前 ------------------
                    elif first_name in ("IF_X", "IF_Z") and second_name == "M":
                        if first_now != now:
                            cbit_idx = cbits_map[(now, first_now)]
                            clbit_obj = new_circ.clbits[cbit_idx]
                            new_circ.measure(second_qs[0], clbit_obj)
                            if first_name == "IF_X":
                                with new_circ.if_test((clbit_obj, 1)):
                                    new_circ.x(first_qs[0])
                            elif first_name == "IF_Z":
                                with new_circ.if_test((clbit_obj, 1)):
                                    new_circ.rz(np.pi / 2, first_qs[0])
                        else:
                            new_circ.measure(second_qs[0], second_qs[0])
                            if first_name == "IF_X":
                                new_circ.x(first_qs[0])
                            elif first_name == "IF_Z":
                                new_circ.rz(np.pi / 2, first_qs[0])

                    indices[now] += 1
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

    
num_qubits = 6
qc = DQCCircuit(num_qubits, num_qubits)

# Build circuit
qc.h(0)
qc.cx(0,1)
qc.cx(1,2)
qc.cx(2,3)
qc.cx(3,4)
qc.cx(4,5)

# Measure all qubits
for i in range(num_qubits):
    qc.measure(i,i)

# Partition and QPUs
Partition = [2,2,2]
qpu1 = DQCQPU(0, "FakeVigoV2")
qpu2 = DQCQPU(1, "FakeAthensV2")
qpu3 = DQCQPU(2, "FakeLagosV2")

# Execute distributed circuit
result_qc = qc.Execution(Partition, [qpu1, qpu2, qpu3])

# Create simulator
sim = AerSimulator()

# Transpile and run
compiled = transpile(result_qc, sim)
job = sim.run(compiled, shots=1000)
result = job.result()

# Get measurement counts
counts = result.get_counts()

# Keep only first 6 bits
new_counts = {}
for bitstring, cnt in counts.items():
    bits = bitstring[:6]
    new_counts[bits] = new_counts.get(bits, 0) + cnt

# Save plots

# 1️⃣ Measurement histogram
plt.figure(figsize=(6, 4))
plot_histogram(new_counts)
plt.title("Measurement Results")
plt.savefig("histogram.png", dpi=300, bbox_inches='tight')
plt.close()

# 2️⃣ Original circuit diagram
fig1 = qc.draw("mpl", scale=0.5)
fig1.savefig("original_circuit.png", dpi=300, bbox_inches='tight')
plt.close(fig1)

# 3️⃣ Merged/executed circuit diagram
fig2 = result_qc.draw("mpl", scale=0.5)
fig2.savefig("result_circuit.png", dpi=300, bbox_inches='tight')
plt.close(fig2)

print("✅ Three figures saved: histogram.png, original_circuit.png, result_circuit.png")

# debug
# for i, tc in enumerate(result_qc):
#     for i, (inst, qargs, cargs) in enumerate(result_qc.data):
#         print(f"Step {i}:")
#         print("  Instruction:", inst.name)

#         # 如果是你自定义的占位符 Instruction
#         if hasattr(inst, "index"):
#             print(f"  [Custom Placeholder] index = {inst.index}")

#         if hasattr(inst, "target"):
#             print(f"  [Custom Placeholder] target = {inst.target}")

#         # 打印 qubit 对应的全局索引
#         if qargs:
#             qubit_indices = [result_qc.find_bit(q).index for q in qargs]
#             print("  Qubits (circuit indices):", qubit_indices)
#             print(qargs)

#         # 打印 clbit 对应的全局索引
#         if cargs:
#             clbit_indices = [result_qc.find_bit(c).index for c in cargs]
#             print("  Clbits (circuit indices):", clbit_indices)

