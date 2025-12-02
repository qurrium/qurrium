"""Declaration of the input fields of Analysis (:mod:`qurry.qurrium.analysis.declare`)"""

from typing import Optional, Union, TypedDict, Any, TypeVar


class AnalyzeArgs(TypedDict):
    """Analysis input prototype."""


_RA = TypeVar("_RA", bound=AnalyzeArgs)
"""The type var of :class:`AnalyzeArgs` for
:meth:`~qurry.qurrium.qurrium.QurriumPrototype.analyze`
and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis`.
"""

SpecificAnalyzeArgs = Optional[dict[str, Union[_RA, dict[str, Any], bool]]]
"""The type hint for :meth:`~qurry.qurrium.multimanager.multimanager.analyze` 
and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis`.
"""
