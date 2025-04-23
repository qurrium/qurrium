"""Declaration - Transpile (:mod:`qurry.declare.transpile`)

Arguments for :func:`transpile` from :mod:`qiskit.compiler.transpiler`
"""

from typing import Union, Callable, Any, Optional, TypedDict

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import Layout, CouplingMap, PropertySet
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from qiskit.transpiler.target import Target


class TranspileArgs(TypedDict, total=False):
    """Transpile arguments for :func:`transpile` from :mod:`qiskit.compiler.transpiler`

    - :mod:`qiskit` 2.0.0

    .. code-block:: python
        _CircuitT = TypeVar("_CircuitT", bound=Union[QuantumCircuit, list[QuantumCircuit]])

        def transpile(  # pylint: disable=too-many-return-statements
            circuits: _CircuitT,
            backend: Optional[Backend] = None,
            basis_gates: Optional[list[str]] = None,
            coupling_map: Optional[Union[CouplingMap, list[list[int]]]] = None,
            initial_layout: Optional[Union[Layout, dict, list]] = None,
            layout_method: Optional[str] = None,
            routing_method: Optional[str] = None,
            translation_method: Optional[str] = None,
            scheduling_method: Optional[str] = None,
            dt: Optional[float] = None,
            approximation_degree: Optional[float] = 1.0,
            seed_transpiler: Optional[int] = None,
            optimization_level: Optional[int] = None,
            callback: Optional[Callable[[
                BasePass, DAGCircuit, float, PropertySet, int
            ], Any]] = None,
            output_name: Optional[Union[str, list[str]]] = None,
            unitary_synthesis_method: str = "default",
            unitary_synthesis_plugin_config: Optional[dict] = None,
            target: Optional[Target] = None,
            hls_config: Optional[HLSConfig] = None,
            init_method: Optional[str] = None,
            optimization_method: Optional[str] = None,
            ignore_backend_supplied_default_methods: bool = False,
            num_processes: Optional[int] = None,
            qubits_initially_zero: bool = True,
        ) -> _CircuitT:
        ...

    """

    basis_gates: Optional[list[str]]
    coupling_map: Optional[Union[CouplingMap, list[list[int]]]]
    initial_layout: Optional[Union[Layout, dict, list]]
    layout_method: Optional[str]
    routing_method: Optional[str]
    translation_method: Optional[str]
    scheduling_method: Optional[str]
    dt: Optional[float]
    approximation_degree: Optional[float]
    seed_transpiler: Optional[int]
    optimization_level: Optional[int]
    callback: Optional[Callable[[BasePass, DAGCircuit, float, PropertySet, int], Any]]
    output_name: Optional[Union[str, list[str]]]
    unitary_synthesis_method: str
    unitary_synthesis_plugin_config: Optional[dict]
    target: Optional[Target]
    hls_config: Optional[HLSConfig]
    init_method: Optional[str]
    optimization_method: Optional[str]
    ignore_backend_supplied_default_methods: bool
    qubits_initially_zero: bool
