"""Post Processing - Randomized Measure - Wavefunction Overlap V1
(:mod:`qurry.process.randomized_measure.wavefunction_overlap_v1`)

Reference:
    .. note::
        - Statistical correlations between locally randomized measurements:
        A toolbox for probing entanglement in many-body quantum states -
        A. Elben, B. Vermersch, C. F. Roos, and P. Zoller,
        [PhysRevA.99.052323](
            https://doi.org/10.1103/PhysRevA.99.052323
        )

    .. code-block:: bibtex
        @article{PhysRevA.99.052323,
            title = {Statistical correlations between locally randomized measurements:
            A toolbox for probing entanglement in many-body quantum states},
            author = {Elben, A. and Vermersch, B. and Roos, C. F. and Zoller, P.},
            journal = {Phys. Rev. A},
            volume = {99},
            issue = {5},
            pages = {052323},
            numpages = {12},
            year = {2019},
            month = {May},
            publisher = {American Physical Society},
            doi = {10.1103/PhysRevA.99.052323},
            url = {https://link.aps.org/doi/10.1103/PhysRevA.99.052323}
        }

"""

from .wavefunction_overlap import (
    randomized_overlap_echo_v1,
    PostProcessingBackendLabel,
    DEFAULT_PROCESS_BACKEND,
)
