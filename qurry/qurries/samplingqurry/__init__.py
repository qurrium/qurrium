"""SamplingExecuter (:mod:`qurry.qurries.samplingqurry`)

- Short name: `sampling_executer`
- Abbreviation: `SE`
"""

from .experiment import SEExperiment
from .arguments import SEMeasureArgs as SamplingExecuterMeasureArgs
from .analysis import DummyAnalysis
from .qurry import QurryV14 as SamplingExecuter
