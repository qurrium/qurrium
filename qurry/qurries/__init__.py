"""Qurries - Other Qurrium Modules (:mod:`qurry.qurries`)

- TwistedOperator - Twisted Operator (Proposal)
    - Formerly known as `qurtwistop`
"""

from .samplingqurry import SamplingExecuter, SamplingExecuterMeasureArgs
from .wavesqurry import WavesExecuter, WavesExecuterMeasureArgs
from .magnet_square import MagnetSquare, MagnetSquareMeasureArgs
from .magnet_square_z import ZDirMagnetSquare, ZDirMagnetSquareMeasureArgs
from .string_operator import StringOperator, StringOperatorMeasureArgs

# from .twisted_operator import TwistedOperator
# from .position_distribution import PositionDistribution
