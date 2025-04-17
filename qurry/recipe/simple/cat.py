"""GHZ state (:mod:`qurry.recipe.library.simple.cat`)

The entangled circuit :cls:`GHZ` a.k.a. :cls:`Cat`.

Reference:
    .. note::
        - Measurement of the Entanglement Spectrum of a Symmetry-Protected Topological State
        Using the IBM Quantum Computer - Choo, Kenny and von Keyserlingk, Curt W. and
        Regnault, Nicolas and Neupert, Titus
        [doi:10.1103/PhysRevLett.121.086808](
            https://doi.org/10.1103/PhysRevLett.121.086808)

    .. code-block:: bibtex
        @article{PhysRevLett.121.086808,
            title = {
                Measurement of the Entanglement Spectrum of a Symmetry-Protected Topological State
                Using the IBM Quantum Computer},
            author = {
                Choo, Kenny and von Keyserlingk, Curt W. and Regnault, Nicolas and Neupert, Titus},
            journal = {Phys. Rev. Lett.},
            volume = {121},
            issue = {8},
            pages = {086808},
            numpages = {5},
            year = {2018},
            month = {Aug},
            publisher = {American Physical Society},
            doi = {10.1103/PhysRevLett.121.086808},
            url = {https://link.aps.org/doi/10.1103/PhysRevLett.121.086808}
}

"""

from ..n_body import OneBody


class GHZ(OneBody):
    r"""The entangled circuit :cls:`GHZ`.
    Introduce in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.086808 .

    .. code-block:: text

        # Open boundary at 8 qubits:
            ┌───┐
        q0: ┤ H ├──■────────────────────────────────
            └───┘┌─┴─┐
        q1: ─────┤ X ├──■───────────────────────────
                 └───┘┌─┴─┐
        q2: ──────────┤ X ├──■──────────────────────
                      └───┘┌─┴─┐
        q3: ───────────────┤ X ├──■─────────────────
                           └───┘┌─┴─┐
        q4: ────────────────────┤ X ├──■────────────
                                └───┘┌─┴─┐
        q5: ─────────────────────────┤ X ├──■───────
                                     └───┘┌─┴─┐
        q6: ──────────────────────────────┤ X ├──■──
                                          └───┘┌─┴─┐
        q7: ───────────────────────────────────┤ X ├
                                               └───┘

    .. math::

        \frac{1}{\sqrt{2}}
            \left({|0\rangle}^{\otimes N} + {|1\rangle}^{\otimes N} \right), N = 8

    Args:
        num_qubits (int): The number of qubits for constructing the example circuit.
        name (str, optional): Name of case. Defaults to "ghz".
    """

    def __init__(self, num_qubits: int, name: str = "ghz") -> None:
        super().__init__(name=name)
        self.num_qubits = num_qubits

    def _build(self) -> None:
        if self._is_built:
            return
        super()._build()

        num_qubits = self.num_qubits
        if num_qubits == 0:
            return

        self.h(0)
        for i in range(1, num_qubits):
            self.cx(i - 1, i)


class Cat(GHZ):
    r""":cls:`Cat`, the anthor name of entangled circuit :cls:`GHZ`.

    .. code-block:: text

        # Open boundary at 8 qubits:
            ┌───┐
        q0: ┤ H ├──■────────────────────────────────
            └───┘┌─┴─┐
        q1: ─────┤ X ├──■───────────────────────────
                 └───┘┌─┴─┐
        q2: ──────────┤ X ├──■──────────────────────
                      └───┘┌─┴─┐
        q3: ───────────────┤ X ├──■─────────────────
                           └───┘┌─┴─┐
        q4: ────────────────────┤ X ├──■────────────
                                └───┘┌─┴─┐
        q5: ─────────────────────────┤ X ├──■───────
                                     └───┘┌─┴─┐
        q6: ──────────────────────────────┤ X ├──■──
                                          └───┘┌─┴─┐
        q7: ───────────────────────────────────┤ X ├
                                               └───┘

    .. math::

        \frac{1}{\sqrt{2}}
            \left({|0\rangle}^{\otimes N} + {|1\rangle}^{\otimes N} \right), N = 8

    Args:
        num_qubits (int): The number of qubits for constructing the example circuit.
        name (str, optional): Name of case. Defaults to "cat".
    """

    def __init__(self, num_qubits: int, name: str = "cat") -> None:
        super().__init__(num_qubits=num_qubits, name=name)
        self.num_qubits = num_qubits
