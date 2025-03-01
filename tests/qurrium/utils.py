"""
================================================================
Circuit Case
================================================================

"""

from typing import Optional, Literal

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.classical import expr

from qurry.recipe.n_body import OneBody, TwoBody
from qurry.tools.datetime import current_time


def cnot_dyn(
    qc: QuantumCircuit,
    control_qubit: int,
    target_qubit: int,
    c1: ClassicalRegister,
    c2: ClassicalRegister,
    add_barriers: Optional[bool] = True,
) -> QuantumCircuit:
    """Generate a CNOT gate bewteen data qubit control_qubit and
    data qubit target_qubit using Bell Pairs.

    Post processing is used to enable the CNOT gate
    via the provided classicial registers c1 and c2

    Assumes that the long-range CNOT gate will be spanning a 1D chain of n-qubits subject
    to nearest-neighbor connections only with the chain starting
    at the control qubit and finishing at the target qubit.

    Assumes that control_qubit < target_qubit (as integers) and
    that the provided circuit qc has |0> set
    qubits control_qubit+1, ..., target_qubit-1

    n = target_qubit - control_qubit - 1 : Number of qubits between the target and control qubits
    k = int(n/2) : Number of Bell pairs created

    .. code-block:: bibtex
        @article{B_umer_2024,
            title={Efficient Long-Range Entanglement Using Dynamic Circuits},
            volume={5},
            ISSN={2691-3399},
            url={http://dx.doi.org/10.1103/PRXQuantum.5.030339},
            DOI={10.1103/prxquantum.5.030339},
            number={3},
            journal={PRX Quantum},
            publisher={American Physical Society (APS)},
            author={
                Bäumer, Elisa and Tripathi, Vinay and Wang, Derek S. and Rall,
                Patrick and Chen, Edward H. and Majumder, Swarnadeep and Seif,
                Alireza and Minev, Zlatko K.},
            year={2024},
            month=aug
        }

    Args:
        qc (QuantumCicruit):
            A Quantum Circuit to add the long range localized unitary CNOT
        control_qubit (int):
            The qubit used as the control.
        target_qubi (int):
            The qubit targeted by the gate.
        c1 (ClassicialRegister):
            Required if n > 1. Register requires k bits
        c2 (ClassicalRegister):
            Required if n > 0. Register requires n - k bits
        add_barriers (bool, optional):
            Default = True. Include barriers before and after long range CNOT

    Note: This approached uses two if_test statements. A better (more performant) approach is
    to have the parity values combined into a single classicial register and then use a switch
    statement. This was done in the associated paper my modifying the qasm file directly.
    The ability to use a switch statement via Qiakit in this way is a future release capability.

    Returns:
        QuantumCircuit
    """
    assert target_qubit > control_qubit
    n = target_qubit - control_qubit - 1
    t = int(n / 2)

    if add_barriers is True:
        qc.barrier()

    # Deteremine where to start the bell pairs and
    # add an extra CNOT when n is odd
    if n % 2 == 0:
        x0 = 1
    else:
        x0 = 2
        qc.cx(0, 1)

    # Create t Bell pairs
    for i in range(t):
        qc.h(x0 + 2 * i)
        qc.cx(x0 + 2 * i, x0 + 2 * i + 1)

    # Entangle Bell pairs and data qubits and measure
    for i in range(t + 1):
        qc.cx(x0 - 1 + 2 * i, x0 + 2 * i)

    for i in range(1, t + x0):
        if i == 1:
            qc.h(2 * i + 1 - x0)
            qc.measure(2 * i + 1 - x0, c2[i - 1])
            parity_control = expr.lift(c2[i - 1])
        else:
            qc.h(2 * i + 1 - x0)
            qc.measure(2 * i + 1 - x0, c2[i - 1])
            parity_control = expr.bit_xor(c2[i - 1], parity_control)

    for i in range(t):
        if i == 0:
            qc.measure(2 * i + x0, c1[i])
            parity_target = expr.lift(c1[i])
        else:
            qc.measure(2 * i + x0, c1[i])
            parity_target = expr.bit_xor(c1[i], parity_target)

    if n > 0:
        with qc.if_test(parity_control):  # type: ignore
            qc.z(0)

    if n > 1:
        with qc.if_test(parity_target):  # type: ignore
            qc.x(-1)

    if add_barriers is True:
        qc.barrier()
    return qc


class CNOTDynCase4To8(OneBody):
    """CNOTDynCase4To8: A circuit with 4 to 8 qubits and a CNOT gate
    between first qubits and last using Bell pairs.

    Or provide a comparison with the normal CNOT gate.

    The circuit is used to provide a test case for multiple classical registers.

    ### The dynamic CNOT gate is used to entangle the first and last qubits

    .. code-block:: text
        # At 6 qubits with 2 classical registers:
             ┌───┐ ░                                                            »
        q_0: ┤ H ├─░─────────────■──────────────────────────────────────────────»
             └───┘ ░ ┌───┐     ┌─┴─┐     ┌─┐                                    »
        q_1: ──────░─┤ H ├──■──┤ X ├─────┤M├────────────────────────────────────»
                   ░ └───┘┌─┴─┐└───┘┌───┐└╥┘   ┌─┐                              »
        q_2: ──────░──────┤ X ├──■──┤ H ├─╫────┤M├──────────────────────────────»
                   ░ ┌───┐└───┘┌─┴─┐└───┘ ║ ┌─┐└╥┘                              »
        q_3: ──────░─┤ H ├──■──┤ X ├──────╫─┤M├─╫───────────────────────────────»
                   ░ └───┘┌─┴─┐└───┘┌───┐ ║ └╥┘ ║ ┌─┐                           »
        q_4: ──────░──────┤ X ├──■──┤ H ├─╫──╫──╫─┤M├───────────────────────────»
                   ░      └───┘┌─┴─┐└───┘ ║  ║  ║ └╥┘┌──────────────────── ┌───┐»
        q_5: ──────░───────────┤ X ├──────╫──╫──╫──╫─┤ If-0 c4[1] ^ c4[0]  ┤ X ├»
                   ░           └───┘      ║  ║  ║  ║ └─────────╥────────── └───┘»
                                          ║  ║  ║  ║       ┌───╨────┐           »
        c4: 2/════════════════════════════╩══╩══╬══╬═══════╡ [expr] ╞═══════════»
                                          0  1  ║  ║       └────────┘           »
        c5: 2/══════════════════════════════════╩══╩════════════════════════════»
                                                0  1                            »

        «                ┌──────────────────── ┌───┐ ───────┐  ░
        « q_0: ──────────┤ If-0 c5[1] ^ c5[0]  ┤ Z ├  End-0 ├──░─
        «                └─────────╥────────── └───┘ ───────┘  ░
        « q_1: ────────────────────╫───────────────────────────░─
        «                          ║                           ░
        « q_2: ────────────────────╫───────────────────────────░─
        «                          ║                           ░
        « q_3: ────────────────────╫───────────────────────────░─
        «                          ║                           ░
        « q_4: ────────────────────╫───────────────────────────░─
        «       ───────┐           ║                           ░
        « q_5:   End-0 ├───────────╫───────────────────────────░─
        «       ───────┘           ║                           ░
        «c4: 2/════════════════════╬═════════════════════════════
        «                      ┌───╨────┐
        «c5: 2/════════════════╡ [expr] ╞════════════════════════
        «                      └────────┘

    ### The comparison CNOT gate is used to entangle the first and last qubits

    .. code-block:: text
        # At 6 qubits:
             ┌───┐
        q_0: ┤ H ├──■──
             └───┘  │
        q_1: ───────┼──
                    │
        q_2: ───────┼──
                    │
        q_3: ───────┼──
                    │
        q_4: ───────┼──
                  ┌─┴─┐
        q_5: ─────┤ X ├
                  └───┘
    """

    @property
    def export(self) -> Literal["dynamic", "comparison"]:
        """The state of the circuit.

        Returns:
            The state of the circuit.
        """
        return self._export

    @export.setter
    def export(self, export: Literal["dynamic", "comparison"]) -> None:
        """Set the state of the circuit.

        Args:
            state: The new state of the circuit.
        """
        if export not in ["dynamic", "comparison"]:
            raise ValueError("Export must be either 'dynamic' or 'comparison'")
        if hasattr(self, "_export"):
            raise AttributeError("Attribute 'export' is read-only.")
        self._export: Literal["dynamic", "comparison"] = export

    def __init__(self, num_qubits, export: Literal["dynamic", "comparison"] = "dynamic") -> None:
        if num_qubits < 4 or num_qubits > 8:
            raise ValueError("Number of qubits must be between 4 and 8")
        super().__init__()
        self.num_qubits = num_qubits
        self.export = export

    def _build(self) -> None:
        if self._is_built:
            return
        super()._build()

        self.h(0)

        if self.export == "dynamic":
            control_qubit = 0
            target_qubit = self.num_qubits - 1
            n = target_qubit - control_qubit - 1
            # Number of qubits between the target and control qubits
            k = int(n / 2)
            # Number of Bell pairs created

            c1 = ClassicalRegister(k, "c1")
            c2 = ClassicalRegister(n - k, "c2")
            self.add_register(c1, c2)

            cnot_dyn(self, control_qubit, target_qubit, c1, c2, add_barriers=True)

        else:
            self.cx(0, self.num_qubits - 1)


class DummyTwoBodyWithDedicatedClbits(TwoBody):
    """DummyTwoBodyWithDedicatedClbit:
    A dummy circuit to simulate a two-body interaction
    with dedicated classical bits and reset gate.
    But the last 2 qubits will be not measured and reset.

    For :cls:`EntropyMeasure` and :cls:`WaveFunctionOverlap` a.k.a. :cls:`EchoListen`.
    It's a product state for no any entanglement.

    To simulate a circuit with its own classical bits as an unit test case for qurry.

    ### The dummy circuit with dedicated classical bits and reset gate

    .. code-block:: text
        # At 6 qubits with 2 classical registers:
              ┌───┐ ░ ┌─┐          ░
        q1_0: ┤ X ├─░─┤M├──────────░──|0>─
              └───┘ ░ └╥┘┌─┐       ░
        q1_1: ──────░──╫─┤M├───────░──|0>─
              ┌───┐ ░  ║ └╥┘┌─┐    ░
        q1_2: ┤ X ├─░──╫──╫─┤M├────░──|0>─
              └───┘ ░  ║  ║ └╥┘┌─┐ ░
        q1_3: ──────░──╫──╫──╫─┤M├─░──|0>─
              ┌───┐ ░  ║  ║  ║ └╥┘ ░
        q1_4: ┤ X ├─░──╫──╫──╫──╫──░──────
              └───┘ ░  ║  ║  ║  ║  ░
        q1_5: ──────░──╫──╫──╫──╫──░──────
                    ░  ║  ║  ║  ║  ░
        c2: 4/═════════╩══╩══╬══╬═════════
                       0  1  ║  ║
        c3: 4/═══════════════╩══╩═════════
                             1  2

    """

    @property
    def clbit_num_cluster(self) -> int:
        """The classical bit number of the cluster for each 2 qubits.

        Returns:
            The classical bit number of the cluster for each 2 qubits.
        """
        return self._clbit_num_cluster

    @clbit_num_cluster.setter
    def clbit_num_cluster(self, clbit_num_cluster: int) -> None:
        """Set the classical bit number of the cluster for each 2 qubits.

        Args:
            clbit_num_cluster (int): The new classical bit number of the cluster for each 2 qubits.

        """
        if hasattr(self, "_clbit_num_cluster"):
            raise AttributeError("Attribute 'clbit_num_cluster' is read-only.")
        self._clbit_num_cluster = clbit_num_cluster

    def __init__(self, num_qubits, clbit_num_cluster: int = 4) -> None:
        if num_qubits % 2 != 0:
            raise ValueError("Number of qubits must be even number")
        if num_qubits < 4:
            raise ValueError(
                "Number of qubits must be greater than 4, "
                + "otherwise it will not any classical bits introduced."
            )
        if clbit_num_cluster < 1:
            raise ValueError("Number of classical bits must be greater than 0")
        super().__init__()
        self.num_qubits = num_qubits
        self.clbit_num_cluster = clbit_num_cluster

    def _build(self) -> None:
        if self._is_built:
            return
        super()._build()

        for i in range(0, self.num_qubits, 2):
            self.x(i)

        self.barrier()

        for i in range(0, self.num_qubits - 2, 2):
            tmp_c = ClassicalRegister(self.clbit_num_cluster, f"c{i}")
            self.add_register(tmp_c)

            self.measure(i, tmp_c[(0 + int(i / 2)) % self.clbit_num_cluster])
            self.measure(i + 1, tmp_c[(1 + int(i / 2)) % self.clbit_num_cluster])

        self.barrier()

        for i in range(0, self.num_qubits - 2):
            self.reset(i)


def current_time_filename():
    """Returns the current time as a filename.

    Returns:
        str: The current time as a filename.
    """
    return current_time().replace(":", "").replace("-", "").replace(" ", "_")
