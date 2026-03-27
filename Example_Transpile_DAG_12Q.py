from collections import Counter, defaultdict
from statistics import mean
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
        }
    return {
        "count": len(values),
        "sum": float(sum(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
        "min": float(min(values)),
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


def circuit_metrics(circ: QuantumCircuit) -> dict:
    """Basic circuit-level metrics."""
    ops = circ.count_ops()
    two_qubit_gates = sum(
        1
        for inst in circ.data
        if len(inst.qubits) == 2 and inst.operation.name != "barrier"
    )

    measure_count = sum(1 for inst in circ.data if inst.operation.name == "measure")
    one_qubit_gates = sum(
        1
        for inst in circ.data
        if len(inst.qubits) == 1 and inst.operation.name not in ("measure", "barrier")
    )

    return {
        "num_qubits": circ.num_qubits,
        "num_clbits": circ.num_clbits,
        "size": circ.size(),
        "depth": circ.depth(),
        "width": circ.width(),
        "num_nonlocal_gates": circ.num_nonlocal_gates(),
        "one_qubit_gate_count": one_qubit_gates,
        "two_qubit_gate_count": two_qubit_gates,
        "measure_count": measure_count,
        "op_histogram": dict(ops),
    }


def dag_metrics(dag) -> dict:
    """Extract DAG-related structural metrics."""
    op_nodes = list(dag.op_nodes())
    op_name_counter = Counter(node.op.name for node in op_nodes)

    qubit_activity = Counter()
    for node in op_nodes:
        for q in node.qargs:
            qubit_activity[str(q)] += 1

    longest_path_nodes = [node for node in dag.longest_path() if isinstance(node, DAGOpNode)]
    longest_path_ops = [node.op.name for node in longest_path_nodes]
    critical_path_2q_count = sum(1 for node in longest_path_nodes if len(node.qargs) == 2)
    critical_path_1q_count = sum(
        1
        for node in longest_path_nodes
        if len(node.qargs) == 1 and node.op.name != "measure"
    )
    critical_path_measure_count = sum(1 for node in longest_path_nodes if node.op.name == "measure")

    layer_two_qubit_counts = []
    layer_total_op_counts = []

    for layer in dag.layers():
        layer_dag = layer["graph"]
        layer_ops = list(layer_dag.op_nodes())
        cnt_2q = sum(1 for n in layer_ops if len(n.qargs) == 2)
        layer_two_qubit_counts.append(cnt_2q)
        layer_total_op_counts.append(len(layer_ops))

    busy_values = list(qubit_activity.values())

    return {
        "dag_depth": dag.depth(),
        "dag_size": dag.size(),
        "num_dag_op_nodes": len(op_nodes),
        "dag_op_histogram": dict(op_name_counter),
        "active_wires": len(list(dag.wires)),
        "longest_path_op_count": len(longest_path_ops),
        "longest_path_ops_preview": longest_path_ops[:30],
        "critical_path_1q_count": critical_path_1q_count,
        "critical_path_2q_count": critical_path_2q_count,
        "critical_path_measure_count": critical_path_measure_count,
        "critical_path_ratio": len(longest_path_ops) / dag.size() if dag.size() else 0.0,
        "max_2q_gates_in_one_layer": max(layer_two_qubit_counts) if layer_two_qubit_counts else 0,
        "avg_2q_gates_per_layer": (
            sum(layer_two_qubit_counts) / len(layer_two_qubit_counts)
            if layer_two_qubit_counts
            else 0.0
        ),
        "avg_ops_per_layer": (
            sum(layer_total_op_counts) / len(layer_total_op_counts)
            if layer_total_op_counts
            else 0.0
        ),
        "parallelism_score": dag.size() / dag.depth() if dag.depth() else 0.0,
        "qubit_activity_stats": safe_stats(busy_values),
        "top_busy_qubits": qubit_activity.most_common(8),
    }


def noise_time_metrics(circ: QuantumCircuit, backend) -> dict:
    """
    Extract noise- and time-related features from a transpiled circuit.
    Assumes `circ` is already transpiled for `backend`.
    """
    target = backend.target

    one_q_errors = []
    two_q_errors = []
    meas_errors = []

    one_q_durations = []
    two_q_durations = []
    meas_durations = []

    used_physical_qubits = set()
    used_2q_edges = Counter()
    op_error_records = []
    op_duration_records = []

    qubit_gate_time = defaultdict(float)
    qubit_gate_error_sum = defaultdict(float)
    qubit_op_count = defaultdict(int)

    swap_count = 0

    for inst in circ.data:
        op = inst.operation
        name = op.name

        if name == "barrier":
            continue

        if name == "swap":
            swap_count += 1

        qargs = [circ.find_bit(q).index for q in inst.qubits]

        for q in qargs:
            used_physical_qubits.add(q)
            qubit_op_count[q] += 1

        err, dur = get_inst_props(target, name, qargs)

        if err is not None:
            op_error_records.append(err)
        if dur is not None:
            op_duration_records.append(dur)

        if dur is not None:
            for q in qargs:
                qubit_gate_time[q] += dur

        if err is not None:
            for q in qargs:
                qubit_gate_error_sum[q] += err

        if name == "measure":
            if err is not None:
                meas_errors.append(err)
            if dur is not None:
                meas_durations.append(dur)

        elif len(qargs) == 1:
            if err is not None:
                one_q_errors.append(err)
            if dur is not None:
                one_q_durations.append(dur)

        elif len(qargs) == 2:
            if err is not None:
                two_q_errors.append(err)
            if dur is not None:
                two_q_durations.append(dur)
            used_2q_edges[tuple(qargs)] += 1

    # Estimated circuit duration
    try:
        estimated_duration = circ.estimate_duration(target=target)
    except Exception:
        estimated_duration = None

    # T1 / T2
    t1_values = []
    t2_values = []
    per_qubit_noise_time = {}

    for q in sorted(used_physical_qubits):
        try:
            qp = backend.qubit_properties(q)
        except Exception:
            qp = None

        t1 = getattr(qp, "t1", None) if qp is not None else None
        t2 = getattr(qp, "t2", None) if qp is not None else None

        if t1 is not None:
            t1_values.append(t1)
        if t2 is not None:
            t2_values.append(t2)

        acc_time = qubit_gate_time.get(q, 0.0)
        acc_err = qubit_gate_error_sum.get(q, 0.0)

        per_qubit_noise_time[q] = {
            "t1": t1,
            "t2": t2,
            "acc_gate_time": acc_time,
            "acc_gate_error_sum": acc_err,
            "op_count": qubit_op_count.get(q, 0),
            "time_over_t1": (acc_time / t1) if (t1 is not None and t1 > 0) else None,
            "time_over_t2": (acc_time / t2) if (t2 is not None and t2 > 0) else None,
        }

    time_over_t1 = [
        x["time_over_t1"]
        for x in per_qubit_noise_time.values()
        if x["time_over_t1"] is not None
    ]
    time_over_t2 = [
        x["time_over_t2"]
        for x in per_qubit_noise_time.values()
        if x["time_over_t2"] is not None
    ]

    qubit_gate_time_values = [per_qubit_noise_time[q]["acc_gate_time"] for q in per_qubit_noise_time]
    qubit_gate_err_values = [per_qubit_noise_time[q]["acc_gate_error_sum"] for q in per_qubit_noise_time]

    return {
        "estimated_duration_sec": estimated_duration,
        "swap_count": swap_count,

        "used_physical_qubits": sorted(used_physical_qubits),
        "num_used_physical_qubits": len(used_physical_qubits),
        "used_2q_edges_top": used_2q_edges.most_common(10),

        "all_op_error_stats": safe_stats(op_error_records),
        "all_op_duration_stats": safe_stats(op_duration_records),

        "one_q_error_stats": safe_stats(one_q_errors),
        "two_q_error_stats": safe_stats(two_q_errors),
        "meas_error_stats": safe_stats(meas_errors),

        "one_q_duration_stats": safe_stats(one_q_durations),
        "two_q_duration_stats": safe_stats(two_q_durations),
        "meas_duration_stats": safe_stats(meas_durations),

        "t1_stats": safe_stats(t1_values),
        "t2_stats": safe_stats(t2_values),

        "time_over_t1_stats": safe_stats(time_over_t1),
        "time_over_t2_stats": safe_stats(time_over_t2),

        "per_qubit_gate_time_stats": safe_stats(qubit_gate_time_values),
        "per_qubit_gate_error_stats": safe_stats(qubit_gate_err_values),

        "per_qubit_noise_time": per_qubit_noise_time,
    }


def build_feature_groups(
    circuit_metrics: dict,
    dag_info: dict,
    noise_time_info: dict,
) -> dict:
    group1_compiled_circuit_structure = {
        "trans_num_qubits": circuit_metrics["num_qubits"],
        "trans_num_clbits": circuit_metrics["num_clbits"],
        "trans_size": circuit_metrics["size"],
        "trans_depth": circuit_metrics["depth"],
        "trans_width": circuit_metrics["width"],
        "trans_num_nonlocal_gates": circuit_metrics["num_nonlocal_gates"],
        "trans_one_qubit_gate_count": circuit_metrics["one_qubit_gate_count"],
        "trans_two_qubit_gate_count": circuit_metrics["two_qubit_gate_count"],
        "trans_measure_count": circuit_metrics["measure_count"],
        "trans_two_qubit_gate_fraction": (
            circuit_metrics["two_qubit_gate_count"] / circuit_metrics["size"]
            if circuit_metrics["size"] else 0.0
        ),

        "dag_depth": dag_info["dag_depth"],
        "dag_size": dag_info["dag_size"],
        "num_dag_op_nodes": dag_info["num_dag_op_nodes"],
        "active_wires": dag_info["active_wires"],
        "longest_path_op_count": dag_info["longest_path_op_count"],
        "critical_path_1q_count": dag_info["critical_path_1q_count"],
        "critical_path_2q_count": dag_info["critical_path_2q_count"],
        "critical_path_measure_count": dag_info["critical_path_measure_count"],
        "critical_path_ratio": dag_info["critical_path_ratio"],
        "max_2q_gates_in_one_layer": dag_info["max_2q_gates_in_one_layer"],
        "avg_2q_gates_per_layer": dag_info["avg_2q_gates_per_layer"],
        "avg_ops_per_layer": dag_info["avg_ops_per_layer"],
        "parallelism_score": dag_info["parallelism_score"],
        "qubit_activity_mean": dag_info["qubit_activity_stats"]["mean"],
        "qubit_activity_max": dag_info["qubit_activity_stats"]["max"],
    }

    group2_layout_routing = {
        "swap_count": noise_time_info["swap_count"],
        "num_used_physical_qubits": noise_time_info["num_used_physical_qubits"],
    }

    group3_hardware_noise = {
        "one_q_error_mean": noise_time_info["one_q_error_stats"]["mean"],
        "one_q_error_max": noise_time_info["one_q_error_stats"]["max"],
        "one_q_error_sum": noise_time_info["one_q_error_stats"]["sum"],

        "two_q_error_mean": noise_time_info["two_q_error_stats"]["mean"],
        "two_q_error_max": noise_time_info["two_q_error_stats"]["max"],
        "two_q_error_sum": noise_time_info["two_q_error_stats"]["sum"],

        "meas_error_mean": noise_time_info["meas_error_stats"]["mean"],
        "meas_error_max": noise_time_info["meas_error_stats"]["max"],
        "meas_error_sum": noise_time_info["meas_error_stats"]["sum"],

        "all_op_error_mean": noise_time_info["all_op_error_stats"]["mean"],
        "all_op_error_max": noise_time_info["all_op_error_stats"]["max"],
        "all_op_error_sum": noise_time_info["all_op_error_stats"]["sum"],

        "t1_mean_used": noise_time_info["t1_stats"]["mean"],
        "t1_min_used": noise_time_info["t1_stats"]["min"],
        "t2_mean_used": noise_time_info["t2_stats"]["mean"],
        "t2_min_used": noise_time_info["t2_stats"]["min"],

        "per_qubit_gate_error_mean": noise_time_info["per_qubit_gate_error_stats"]["mean"],
        "per_qubit_gate_error_max": noise_time_info["per_qubit_gate_error_stats"]["max"],
    }

    group4_time_decoherence = {
        "estimated_duration_sec": (
            noise_time_info["estimated_duration_sec"]
            if noise_time_info["estimated_duration_sec"] is not None
            else -1.0
        ),

        "one_q_duration_sum": noise_time_info["one_q_duration_stats"]["sum"],
        "two_q_duration_sum": noise_time_info["two_q_duration_stats"]["sum"],
        "meas_duration_sum": noise_time_info["meas_duration_stats"]["sum"],

        "all_op_duration_mean": noise_time_info["all_op_duration_stats"]["mean"],
        "all_op_duration_max": noise_time_info["all_op_duration_stats"]["max"],
        "all_op_duration_sum": noise_time_info["all_op_duration_stats"]["sum"],

        "time_over_t1_mean": noise_time_info["time_over_t1_stats"]["mean"],
        "time_over_t1_max": noise_time_info["time_over_t1_stats"]["max"],
        "time_over_t2_mean": noise_time_info["time_over_t2_stats"]["mean"],
        "time_over_t2_max": noise_time_info["time_over_t2_stats"]["max"],

        "per_qubit_gate_time_mean": noise_time_info["per_qubit_gate_time_stats"]["mean"],
        "per_qubit_gate_time_max": noise_time_info["per_qubit_gate_time_stats"]["max"],
    }

    return {
        "group1_compiled_circuit_structure": group1_compiled_circuit_structure,
        "group2_layout_routing": group2_layout_routing,
        "group3_hardware_noise": group3_hardware_noise,
        "group4_time_decoherence": group4_time_decoherence,
    }

def build_flattened_feature_vector(feature_groups: dict) -> dict:
    flat = {}
    for group_name in [
        "group1_compiled_circuit_structure",
        "group2_layout_routing",
        "group3_hardware_noise",
        "group4_time_decoherence",
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

   # 6) Build grouped features
    feature_groups = build_feature_groups(
        circuit_info,
        dag_info,
        noise_time_info,
    )

    feature_vector = build_flattened_feature_vector(feature_groups)

    pretty_print_dict("Group 1: Compiled Circuit Structure", feature_groups["group1_compiled_circuit_structure"])
    pretty_print_dict("Group 2: Layout / Routing", feature_groups["group2_layout_routing"])
    pretty_print_dict("Group 3: Hardware Noise", feature_groups["group3_hardware_noise"])
    pretty_print_dict("Group 4: Time / Decoherence", feature_groups["group4_time_decoherence"])
    pretty_print_dict("Flattened Feature Vector (for ML)", feature_vector)


if __name__ == "__main__":
    main()