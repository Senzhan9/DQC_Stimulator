from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# -------------------------------------------------
# 1. Build GHZ
# -------------------------------------------------
def build_ghz_circuit(n_qubits: int = 6) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


# -------------------------------------------------
# 2. Same "platform style": fixed basis + fixed coupling map
#    Use a 6-qubit connected chain, so memory stays small.
# -------------------------------------------------
def get_platform_spec(n_qubits: int):
    basis_gates = ["rz", "sx", "x", "ecr"]

    edges = []
    for i in range(n_qubits - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    coupling = CouplingMap(edges)
    return basis_gates, coupling


# -------------------------------------------------
# 3. Build two noise models: low / high
# -------------------------------------------------
def make_noise_model(p1: float, p2: float) -> NoiseModel:
    noise_model = NoiseModel()

    err_1q = depolarizing_error(p1, 1)
    err_2q = depolarizing_error(p2, 2)

    for gate in ["rz", "sx", "x"]:
        noise_model.add_all_qubit_quantum_error(err_1q, gate)

    noise_model.add_all_qubit_quantum_error(err_2q, ["ecr"])

    return noise_model


# -------------------------------------------------
# 4. Noisy density-matrix simulation
# -------------------------------------------------
def simulate_density_matrix(circuit: QuantumCircuit, noise_model: NoiseModel):
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)

    circ = circuit.copy()
    circ.save_density_matrix()

    result = sim.run(circ).result()
    rho = result.data(0)["density_matrix"]
    return rho


# -------------------------------------------------
# 5. Main
# -------------------------------------------------
def main():
    n_qubits = 10
    ghz = build_ghz_circuit(n_qubits)

    basis_gates, coupling = get_platform_spec(n_qubits)

    ghz_t = transpile(
        ghz,
        basis_gates=basis_gates,
        coupling_map=coupling,
        optimization_level=3,
        seed_transpiler=42,
    )

    # Ideal target
    ideal_state = Statevector.from_instruction(ghz_t)

    # Low-noise version
    noise_low = make_noise_model(
        p1=0.0020,
        p2=0.020,
    )

    # High-noise version
    noise_high = make_noise_model(
        p1=0.0035,
        p2=0.020,
    )

    rho_low = simulate_density_matrix(ghz_t, noise_low)
    rho_high = simulate_density_matrix(ghz_t, noise_high)

    fid_low = state_fidelity(ideal_state, rho_low)
    fid_high = state_fidelity(ideal_state, rho_high)

    print("=== Transpiled GHZ circuit ===")
    print(ghz_t)
    print()

    ops = ghz_t.count_ops()

    total_gates = sum(ops.values())
    two_q_gates = sum(v for k, v in ops.items() if k in ["cx", "cz", "swap", "ecr"])
    one_q_gates = total_gates - two_q_gates

    print("num_qubits :", ghz_t.num_qubits)
    print("depth      :", ghz_t.depth())
    print("1Q gates   :", one_q_gates)
    print("2Q gates   :", two_q_gates)
    print("total gates:", total_gates)
    print()

    print("=== Fidelity comparison ===")
    print(f"Low-noise fidelity  : {fid_low:.6f}")
    print(f"High-noise fidelity : {fid_high:.6f}")
    print(f"Δ Fidelity          : {fid_low - fid_high:.6f}")


if __name__ == "__main__":
    main()