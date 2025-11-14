"""Transpiler container (:mod:`qurry.qurrium.container.transpiler`)"""

from typing import Union, Callable, Any, Optional, TypedDict

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import Layout, CouplingMap, PropertySet
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from qiskit.transpiler.target import Target
from qiskit.transpiler.passmanager import PassManager

PassManagerType = Optional[Union[str, PassManager, tuple[str, PassManager]]]
"""The type hint for passmanager argument in 
:meth:`~qurry.qurrium.qurrium.QurriumPrototype.output`."""


class PassManagerContainer(dict[str, PassManager]):
    """A customized dictionary for storing
    :class:`~qiskit.transpiler.passmanager.PassManager` objects."""

    __name__ = "PassManagerContainer"

    def __repr__(self):
        original_repr = repr(dict(self.items()))
        return f"{self.__name__}({original_repr}, num={len(self)})"

    def _repr_oneline(self):
        return f"{self.__name__}(" + "{...}" + f", num={len(self)})"

    def _repr_pretty_(self, p, cycle):
        # pylint: disable=protected-access
        original_repr = repr(dict(self.items()))
        # pylint: enable=protected-access
        original_repr_split = original_repr[1:-1].split(", ")
        length = len(original_repr_split)

        if cycle:
            p.text(f"{self.__name__}(" + "{...}" + f", num={length})")
        else:
            with p.group(2, f"{self.__name__}(num={length}" + ", {", "})"):
                for i, item in enumerate(original_repr_split):
                    p.breakable()
                    p.text(item)
                    if i < length - 1:
                        p.text(",")

    def __str__(self):
        return super().__repr__()


class TranspileArgs(TypedDict, total=False):
    """Transpile arguments for :func:`~qiskit.compiler.transpiler.transpile`

    - :mod:`~qiskit` 2.0.0

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
