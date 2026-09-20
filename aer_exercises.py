"""
Hands-on Qiskit Aer 0.17.1 exercises.
Run the whole file, or comment out sections to isolate one exercise at a time:
    .venv/bin/python aer_exercises.py
"""
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# 1. Single qubit, no entanglement — just flip it
section("1. X gate on |0> — should deterministically give '1'")
qc = QuantumCircuit(1, 1)
qc.x(0)
qc.measure(0, 0)
backend = AerSimulator()
result = backend.run(transpile(qc, backend), shots=1024).result()
print(result.get_counts())


# 2. Superposition — H gate should give ~50/50
section("2. H gate on |0> — should give ~50/50 split")
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
result = backend.run(transpile(qc, backend), shots=1024).result()
print(result.get_counts())


# 3. Entanglement — Bell state, correlated outcomes only
section("3. Bell state (H + CX) — only '00' and '11' should appear")
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
result = backend.run(transpile(qc, backend), shots=1024).result()
print(result.get_counts())


# 4. Inspect the exact statevector (no shots, no randomness)
section("4. Exact statevector of the Bell circuit (pre-measurement)")
qc_no_measure = QuantumCircuit(2)
qc_no_measure.h(0)
qc_no_measure.cx(0, 1)
sv = Statevector(qc_no_measure)
print(sv)
print("Probabilities:", sv.probabilities_dict())


# 5. Inspect the circuit's unitary matrix
section("5. Unitary matrix of H acting on 1 qubit")
qc_h = QuantumCircuit(1)
qc_h.h(0)
op = Operator(qc_h)
print(op.data)


# 6. Compare simulation methods explicitly
section("6. Same Bell circuit, forced through different methods")
for method in ["statevector", "stabilizer", "matrix_product_state"]:
    sim = AerSimulator(method=method)
    result = sim.run(transpile(qc, sim), shots=1024).result()
    print(f"{method:20s} -> {result.get_counts()}")


# 7. Noise model changes the outcome distribution
section("7. Ideal vs noisy Bell state (5% depolarizing error on CX)")
# Noise on h/x alone won't leak into '01'/'10': CX still perfectly
# correlates whatever state qubit 0 is in. Noise on the 2-qubit gate
# itself is what breaks the correlation.
noise_model = NoiseModel()
error_2q = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

ideal = AerSimulator()
noisy = AerSimulator(noise_model=noise_model)

ideal_counts = ideal.run(transpile(qc, ideal), shots=4096).result().get_counts()
noisy_counts = noisy.run(transpile(qc, noisy), shots=4096).result().get_counts()
print("Ideal:", ideal_counts)
print("Noisy:", noisy_counts, "  <- expect some '01'/'10' leaking in")


# 8. Shots vs exact probabilities — sampling noise shrinks with more shots
section("8. Sampling noise shrinks as shots increase")
for shots in [10, 100, 10000]:
    result = backend.run(transpile(qc, backend), shots=shots).result()
    counts = result.get_counts()
    p11 = counts.get("11", 0) / shots
    print(f"shots={shots:6d}  P(11) ~= {p11:.4f}  (exact = 0.5)")
