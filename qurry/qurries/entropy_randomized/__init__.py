"""EntropyMeasureRandomized - The Second Order Renyi Entropy by Randomized Measurement
(:mod:`qurry.qurries.entropy_randomized`)

- Short name: `entropy_randomized`
- Acronym: `EMR`
"""

from .experiment import EMRExperiment
from .tales import EntropyMeasureTales, EntropyMeasureTalesTypes
from .arguments import EMRMeasureArgs, SHORT_NAME, ACRONYM
from .analysis import EMRAnalysis
from .qurry import EntropyMeasureRandomized
