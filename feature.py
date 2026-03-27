from collections import Counter, defaultdict
from statistics import mean, pstdev
from math import pi

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit_ibm_runtime.fake_provider import FakeKolkataV2


def build_common_ansatz(num_qubits: int = 12, layers: int = 3) -> QuantumCircuit:
    """Create a common hardware-efficient ansatz circuit."""
    qc = QuantumCircuit(num_qubits, num_qubits, name="HEA_12Q")

    for q in range(num_qubits):
        qc.h(q)

    for layer in range(layers):
        for q in range(num_qubits):
            theta = (layer + 1) * (q + 1) * pi / (num_qubits + 1)
            phi = (layer + 1) * (q + 2) * pi / (num_qubits + 3)
            qc.ry(theta, q)
            qc.rz(phi, q)

        # Ring entanglement
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(num_qubits - 1, 0)

    qc.barrier()
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def safe_stats(values):
    """Return simple statistics for a numeric list."""
    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "std": 0.0,
        }

    return {
        "count": len(values),
        "sum": float(sum(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
        "min": float(min(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def get_inst_props(target, op_name, qargs):
    """
    Safely read instruction properties from backend.target.
    Returns (error, duration), both may be None.
    """
    try:
        if op_name not in target:
            return None, None
        props_map = target[op_name]
        props = props_map.get(tuple(qargs), None)
    except Exception:
        props = None

    if props is None:
        return None, None

    err = getattr(props, "error", None)
    dur = getattr(props, "duration", None)
    return err, dur


def kraus_comm_noise_strength(op) -> float | None:
    """
    Convert a Kraus instruction into a scalar communication-noise strength.
    Uses: error = 1 - entanglement_fidelity_to_identity.
    """
    try:
        kraus_ops = list(getattr(op, "params", []))
        if not kraus_ops:
            return None

        dim = kraus_ops[0].shape[0]
        if dim <= 0:
            return None

        fe = sum(abs(k.trace()) ** 2 for k in kraus_ops) / (dim * dim)
        err = 1.0 - float(fe.real)
        if err < 0:
            return 0.0
        if err > 1:
            return 1.0
        return err
    except Exception:
        return None


def circuit_metrics(circ: QuantumCircuit) -> dict:
    """Basic circuit-level metrics."""
    two_qubit_gates = sum(
        1
        for inst in circ.data
        if len(inst.qubits) == 2 and inst.operation.name != "barrier"
    )

    measure_count = sum(1 for inst in circ.data if inst.operation.name == "measure")
    comm_count = sum(1 for inst in circ.data if inst.operation.name == "kraus")
    one_qubit_gates = sum(
        1
        for inst in circ.data
        if len(inst.qubits) == 1 and inst.operation.name not in ("measure", "barrier")
    )

    return {
        "num_qubits": circ.num_qubits,
        "depth": circ.depth(),
        "one_qubit_gate_count": one_qubit_gates,
        "two_qubit_gate_count": two_qubit_gates,
        "measure_count": measure_count,
        "comm_count": comm_count,
    }


def dag_metrics(dag) -> dict:
    """Extract only the selected DAG-related features."""
    op_nodes = list(dag.op_nodes())

    qubit_activity = Counter()
    for node in op_nodes:
        for q in node.qargs:
            qubit_activity[str(q)] += 1

    longest_path_nodes = [node for node in dag.longest_path() if isinstance(node, DAGOpNode)]

    critical_path_2q_count = sum(1 for node in longest_path_nodes if len(node.qargs) == 2)
    critical_path_1q_count = sum(
        1
        for node in longest_path_nodes
        if len(node.qargs) == 1 and node.op.name != "measure"
    )

    busy_values = list(qubit_activity.values())
    busy_stats = safe_stats(busy_values)

    return {
        "critical_path_1q_count": critical_path_1q_count,
        "critical_path_2q_count": critical_path_2q_count,
        "critical_path_ratio": len(longest_path_nodes) / dag.size() if dag.size() else 0.0,
        "qubit_activity_mean": busy_stats["mean"],
        "qubit_activity_max": busy_stats["max"],
    }


def noise_time_metrics(circ: QuantumCircuit, backend) -> dict:
    """
    Extract selected noise- and time-related features from a transpiled circuit.
    Assumes `circ` is already transpiled for `backend`.
    """
    target = backend.target

    one_q_errors = []
    two_q_errors = []
    meas_errors = []
    comm_noise_errors = []

    two_q_durations = []

    op_error_records = []

    qubit_gate_time = defaultdict(float)
    qubit_gate_error_sum = defaultdict(float)

    used_physical_qubits = set()

    for inst in circ.data:
        op = inst.operation
        name = op.name

        if name == "barrier":
            continue

        qargs = [circ.find_bit(q).index for q in inst.qubits]

        for q in qargs:
            used_physical_qubits.add(q)

        err, dur = get_inst_props(target, name, qargs)

        if err is not None:
            op_error_records.append(err)
            for q in qargs:
                qubit_gate_error_sum[q] += err

        if dur is not None:
            for q in qargs:
                qubit_gate_time[q] += dur

        if name == "measure":
            if err is not None:
                meas_errors.append(err)

        if name == "kraus":
            comm_err = kraus_comm_noise_strength(op)
            if comm_err is not None:
                comm_noise_errors.append(comm_err)

        elif len(qargs) == 1:
            if err is not None:
                one_q_errors.append(err)

        elif len(qargs) == 2:
            if err is not None:
                two_q_errors.append(err)
            if dur is not None:
                two_q_durations.append(dur)

    # Estimated circuit duration
    try:
        estimated_duration = circ.estimate_duration(target=target)
    except Exception:
        estimated_duration = None

    # T1 / T2 based normalized time
    t1_ratios = []
    t2_ratios = []

    for q in sorted(used_physical_qubits):
        try:
            qp = backend.qubit_properties(q)
        except Exception:
            qp = None

        t1 = getattr(qp, "t1", None) if qp is not None else None
        t2 = getattr(qp, "t2", None) if qp is not None else None
        acc_time = qubit_gate_time.get(q, 0.0)

        if t1 is not None and t1 > 0:
            t1_ratios.append(acc_time / t1)
        if t2 is not None and t2 > 0:
            t2_ratios.append(acc_time / t2)

    per_qubit_gate_err_values = list(qubit_gate_error_sum.values())
    per_qubit_gate_time_values = list(qubit_gate_time.values())

    one_q_stats = safe_stats(one_q_errors)
    two_q_stats = safe_stats(two_q_errors)
    meas_stats = safe_stats(meas_errors)
    comm_noise_stats = safe_stats(comm_noise_errors)
    all_op_error_stats = safe_stats(op_error_records)
    two_q_duration_stats = safe_stats(two_q_durations)
    t1_ratio_stats = safe_stats(t1_ratios)
    t2_ratio_stats = safe_stats(t2_ratios)
    per_qubit_gate_err_stats = safe_stats(per_qubit_gate_err_values)
    per_qubit_gate_time_stats = safe_stats(per_qubit_gate_time_values)

    return {
        # Noise strength: 1Q
        "one_q_error_sum": one_q_stats["sum"],
        "one_q_error_mean": one_q_stats["mean"],
        "one_q_error_max": one_q_stats["max"],
        "one_q_error_std": one_q_stats["std"],

        # Noise strength: 2Q
        "two_q_error_sum": two_q_stats["sum"],
        "two_q_error_mean": two_q_stats["mean"],
        "two_q_error_max": two_q_stats["max"],
        "two_q_error_std": two_q_stats["std"],

        # Noise strength: Measurement
        "meas_error_sum": meas_stats["sum"],
        "meas_error_mean": meas_stats["mean"],
        "meas_error_max": meas_stats["max"],
        "meas_error_std": meas_stats["std"],

        # Noise strength: Communication (Kraus instruction)
        "comm_noise_mean": comm_noise_stats["mean"],
        "comm_noise_max": comm_noise_stats["max"],
        "comm_noise_min": comm_noise_stats["min"],
        "comm_noise_std": comm_noise_stats["std"],

        # All ops
        "all_op_error_sum": all_op_error_stats["sum"],

        # Time / decoherence
        "estimated_duration_sec": (
            estimated_duration if estimated_duration is not None else -1.0
        ),
        "two_q_duration_sum": two_q_duration_stats["sum"],
        "time_over_t1_mean": t1_ratio_stats["mean"],
        "time_over_t1_max": t1_ratio_stats["max"],
        "time_over_t2_mean": t2_ratio_stats["mean"],
        "time_over_t2_max": t2_ratio_stats["max"],

        # Load distribution
        "per_qubit_gate_error_max": per_qubit_gate_err_stats["max"],
        "per_qubit_gate_time_max": per_qubit_gate_time_stats["max"],
    }


def build_feature_groups(
    circuit_info: dict,
    dag_info: dict,
    noise_time_info: dict,
) -> dict:
    # A. 基础结构
    group_a_basic_structure = {
        "num_qubits": circuit_info["num_qubits"],
        "depth": circuit_info["depth"],
        "one_qubit_gate_count": circuit_info["one_qubit_gate_count"],
        "two_qubit_gate_count": circuit_info["two_qubit_gate_count"],
        "measure_count": circuit_info["measure_count"],
        "comm_count": circuit_info["comm_count"],
    }

    # B. 关键路径
    group_b_critical_path = {
        "critical_path_1q_count": dag_info["critical_path_1q_count"],
        "critical_path_2q_count": dag_info["critical_path_2q_count"],
        "critical_path_ratio": dag_info["critical_path_ratio"],
    }

    # C. 噪声强度
    group_c_noise_strength = {
        "one_q_error_sum": noise_time_info["one_q_error_sum"],
        "one_q_error_mean": noise_time_info["one_q_error_mean"],
        "one_q_error_max": noise_time_info["one_q_error_max"],
        "one_q_error_std": noise_time_info["one_q_error_std"],

        "two_q_error_sum": noise_time_info["two_q_error_sum"],
        "two_q_error_mean": noise_time_info["two_q_error_mean"],
        "two_q_error_max": noise_time_info["two_q_error_max"],
        "two_q_error_std": noise_time_info["two_q_error_std"],

        "meas_error_sum": noise_time_info["meas_error_sum"],
        "meas_error_mean": noise_time_info["meas_error_mean"],
        "meas_error_max": noise_time_info["meas_error_max"],
        "meas_error_std": noise_time_info["meas_error_std"],

        "comm_noise_mean": noise_time_info["comm_noise_mean"],
        "comm_noise_max": noise_time_info["comm_noise_max"],
        "comm_noise_min": noise_time_info["comm_noise_min"],
        "comm_noise_std": noise_time_info["comm_noise_std"],

        "all_op_error_sum": noise_time_info["all_op_error_sum"],
    }

    # D. 时间 / 退相干
    group_d_time_decoherence = {
        "estimated_duration_sec": noise_time_info["estimated_duration_sec"],
        "two_q_duration_sum": noise_time_info["two_q_duration_sum"],
        "time_over_t1_mean": noise_time_info["time_over_t1_mean"],
        "time_over_t1_max": noise_time_info["time_over_t1_max"],
        "time_over_t2_mean": noise_time_info["time_over_t2_mean"],
        "time_over_t2_max": noise_time_info["time_over_t2_max"],
    }

    # E. 负载分布
    group_e_load_distribution = {
        "qubit_activity_mean": dag_info["qubit_activity_mean"],
        "qubit_activity_max": dag_info["qubit_activity_max"],
        "per_qubit_gate_error_max": noise_time_info["per_qubit_gate_error_max"],
        "per_qubit_gate_time_max": noise_time_info["per_qubit_gate_time_max"],
    }

    return {
        "group_a_basic_structure": group_a_basic_structure,
        "group_b_critical_path": group_b_critical_path,
        "group_c_noise_strength": group_c_noise_strength,
        "group_d_time_decoherence": group_d_time_decoherence,
        "group_e_load_distribution": group_e_load_distribution,
    }


def build_flattened_feature_vector(feature_groups: dict) -> dict:
    flat = {}
    for group_name in [
        "group_a_basic_structure",
        "group_b_critical_path",
        "group_c_noise_strength",
        "group_d_time_decoherence",
        "group_e_load_distribution",
    ]:
        flat.update(feature_groups[group_name])
    return flat


def pretty_print_dict(title: str, d: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in d.items():
        print(f"{k}: {v}")


def main() -> None:
    # 1) Build original circuit
    qc = build_common_ansatz(num_qubits=12, layers=3)

    # 2) Backend
    backend = FakeKolkataV2()

    # 3) Transpile
    tqc = transpile(
        qc,
        backend=backend,
        optimization_level=3,
        seed_transpiler=7,
    )

    # 4) Convert to DAG
    dag = circuit_to_dag(tqc)

    # 5) Extract metrics
    circuit_info = circuit_metrics(tqc)
    dag_info = dag_metrics(dag)
    noise_time_info = noise_time_metrics(tqc, backend)

    # 6) Build selected feature groups
    feature_groups = build_feature_groups(
        circuit_info,
        dag_info,
        noise_time_info,
    )

    feature_vector = build_flattened_feature_vector(feature_groups)

    pretty_print_dict("A. Basic Structure", feature_groups["group_a_basic_structure"])
    pretty_print_dict("B. Critical Path", feature_groups["group_b_critical_path"])
    pretty_print_dict("C. Noise Strength", feature_groups["group_c_noise_strength"])
    pretty_print_dict("D. Time / Decoherence", feature_groups["group_d_time_decoherence"])
    pretty_print_dict("E. Load Distribution", feature_groups["group_e_load_distribution"])
    pretty_print_dict("Flattened Feature Vector (for ML)", feature_vector)


if __name__ == "__main__":
    main()
