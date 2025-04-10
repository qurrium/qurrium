"""StringOperator - String Operator
(:mod:`qurry.qurries.string_operator`)

Formerly known as `qurstrop`

Reference:
    .. note::
        - Crossing a topological phase transition with a quantum computer -
        Smith, Adam and Jobst, Bernhard and Green, Andrew G. and Pollmann, Frank,
        [PhysRevResearch.4.L022020](https://link.aps.org/doi/10.1103/PhysRevResearch.4.L022020)

    .. code-block:: bibtex
        @article{PhysRevResearch.4.L022020,
            title = {Crossing a topological phase transition with a quantum computer},
            author = {Smith, Adam and Jobst, Bernhard and Green, Andrew G. and Pollmann, Frank},
            journal = {Phys. Rev. Research},
            volume = {4},
            issue = {2},
            pages = {L022020},
            numpages = {8},
            year = {2022},
            month = {Apr},
            publisher = {American Physical Society},
            doi = {10.1103/PhysRevResearch.4.L022020},
            url = {https://link.aps.org/doi/10.1103/PhysRevResearch.4.L022020}
        }

"""

from .analysis import StringOperatorAnalysis
from .experiment import StringOperatorExperiment
from .qurry import StringOperator
