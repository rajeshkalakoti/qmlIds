from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


if __name__ == "__main__":
    backend = AerSimulator()
    qc = bell_circuit()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=1024).result()
    counts = result.get_counts()

    print("Backend:", backend.name)
    print("Counts:", counts)
